import os
import io
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from tqdm import tqdm

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# =========================
# Auth / Service
# =========================
def _get_creds(client_secret_path: str, token_path: str) -> Credentials:
    """
    Returns valid OAuth credentials. Prompts once (browser) on first run.
    Token is cached at token_path for silent refresh afterwards.
    """
    token_file = Path(token_path)
    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            # Localhost callback; browser opens once on first run
            creds = flow.run_local_server(port=0)

        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")

    return creds


def _build_drive(creds: Credentials):
    """
    Build a Drive v3 service client. For thread-safety we create a client per thread.
    """
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# =========================
# Listing
# =========================
def _list_children(drive, folder_id: str) -> List[Dict]:
    """
    Lists immediate children of a folder. Supports My Drive and Shared Drives.
    """
    query = f"'{folder_id}' in parents and trashed = false"
    fields = "nextPageToken, files(id, name, mimeType, md5Checksum)"
    items, page_token = [], None

    while True:
        resp = drive.files().list(
            q=query,
            fields=fields,
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            corpora="allDrives",
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return items


def _is_folder(item: Dict) -> bool:
    return item.get("mimeType") == "application/vnd.google-apps.folder"


# =========================
# File utilities
# =========================
def _md5(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_google_doc_mime(mime: str) -> bool:
    return mime.startswith("application/vnd.google-apps.") and mime != "application/vnd.google-apps.folder"


# =========================
# Walk Drive tree (skip Google Docs)
# =========================
def _walk_drive(drive, root_id: str, rel: Path = Path(".")) -> List[Dict]:
    """
    Recursively walk a Google Drive folder and return a flat list of downloadable entries.
    Skips Google Docs/Sheets/Slides/etc. (not downloadable via get_media).
    """
    results: List[Dict] = []
    stack: List[Tuple[str, Path]] = [(root_id, rel)]
    discovered = 0

    while stack:
        folder_id, relpath = stack.pop()
        children = _list_children(drive, folder_id)
        discovered += len(children)
        if discovered and discovered % 500 == 0:
            print(f"  discovered {discovered} items so far...", flush=True)

        for item in children:
            mime = item.get("mimeType", "")
            name = item.get("name", "")

            if _is_folder(item):
                stack.append((item["id"], relpath / name))
                continue

            # Skip Google Docs/Sheets/Slides/Forms/Drawings/Scripts
            if _is_google_doc_mime(mime):
                # Optional: uncomment to log skips
                # print(f"  [skip-doc] {name} ({mime})")
                continue

            results.append({
                "id": item["id"],
                "name": name,
                "relpath": relpath / name,
                "md5": item.get("md5Checksum"),
                "mime": mime,
            })

    return results


# =========================
# Download (single attempt)
# =========================
def _download_file_once(service, file_id: str, dest_path: Path, chunk_mb: int = 8) -> None:
    """
    Download a file to dest_path atomically using a .part temporary file.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    request = service.files().get_media(fileId=file_id)
    chunk_size = max(256 * 1024, int(chunk_mb * 1024 * 1024))  # >= 256 KB
    downloader = MediaIoBaseDownload(io.FileIO(tmp_path, "wb"), request, chunksize=chunk_size)

    done = False
    while not done:
        _status, done = downloader.next_chunk()

    # Atomic replace
    if dest_path.exists():
        dest_path.unlink()
    tmp_path.rename(dest_path)


# =========================
# Download with retries / backoff
# =========================
def _download_with_retries(
    creds: Credentials,
    file_id: str,
    dest_path: Path,
    expected_md5: Optional[str],
    chunk_mb: int = 8,
    max_retries: int = 5,
    backoff_base: float = 0.8,
) -> Tuple[str, Optional[str]]:
    """
    Returns (status, message):
      status in {"ok", "skipped", "failed"}
      message optional details
    """
    # Build a thread-local Drive client
    service = _build_drive(creds)

    for attempt in range(1, max_retries + 1):
        try:
            _download_file_once(service, file_id, dest_path, chunk_mb=chunk_mb)

            if expected_md5:
                local_md5 = _md5(dest_path)
                if local_md5 != expected_md5:
                    # Bad checksum → retry
                    if dest_path.exists():
                        dest_path.unlink()
                    raise IOError(f"md5 mismatch: got {local_md5}, want {expected_md5}")

            return "ok", None

        except HttpError as e:
            # Not downloadable (Google Docs) → skip (should have been filtered earlier)
            if e.resp.status == 403:
                # Keep "fileNotDownloadable" as a skip
                try:
                    reason_text = e.error_details if hasattr(e, "error_details") else str(e)
                except Exception:
                    reason_text = str(e)
                if "fileNotDownloadable" in str(e) or "use Export" in str(e).lower():
                    return "skipped", "google-docs-not-downloadable"

            # Rate limiting / transient errors: 429 or 5xx
            if e.resp.status in (429, 500, 502, 503, 504):
                sleep_s = backoff_base * (2 ** (attempt - 1))
                time.sleep(sleep_s)
                continue

            # Other 4xx: fail fast
            return "failed", f"http {e.resp.status}"

        except Exception as ex:
            # Transient I/O errors: backoff and retry
            if attempt < max_retries:
                sleep_s = backoff_base * (2 ** (attempt - 1))
                time.sleep(sleep_s)
                continue
            return "failed", str(ex)

    return "failed", "max-retries-exceeded"


# =========================
# Public API
# =========================
def ensure_local_tree(
    folder_id: str,
    cache_root: str = r"C:\Data\project_cache",
    client_secret_path: str = r"C:\Dev\gpu\client_secret.json",
    token_path: str = os.path.join(os.path.expanduser("~"), ".credentials", "gdrive_token.json"),
    selective_globs: Optional[List[str]] = None,
    max_workers: int = 8,
    chunk_mb: int = 8,
) -> str:
    """
    Recursively mirror a Drive folder to a local cache folder.
    - Skips native Google Docs types (Docs/Sheets/Slides/etc.).
    - Downloads in parallel (max_workers).
    - Verifies MD5 when available.
    - Retries transient errors with exponential backoff.

    Args
    ----
    folder_id : str
        Google Drive folder ID to sync from.
    cache_root : str
        Local root directory for the mirror (created if missing).
    client_secret_path : str
        Path to your OAuth client_secret.json.
    token_path : str
        Path to store the user's OAuth token JSON.
    selective_globs : list[str] | None
        Optional list of glob patterns to restrict which files sync (e.g., ["data/**", "**/*.csv"]).
    max_workers : int
        Number of parallel download workers (default 8). Tune based on bandwidth/IT policy.
    chunk_mb : int
        Chunk size (in MB) for each download request (default 8MB).

    Returns
    -------
    str
        Absolute path to the local cache root.
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    creds = _get_creds(client_secret_path, token_path)
    # Use a single service for listing (safe on main thread)
    listing_service = _build_drive(creds)

    # 1) Discover files
    print(f"Listing Google Drive folder: {folder_id}")
    files = _walk_drive(listing_service, folder_id)

    # 2) Optional filtering
    if selective_globs:
        from fnmatch import fnmatch
        def keep(p: Path) -> bool:
            s = str(p).replace("\\", "/")
            return any(fnmatch(s, g) for g in selective_globs)
        files = [f for f in files if keep(f["relpath"])]

    # 3) Build to-do list based on MD5 (or presence if MD5 missing)
    to_download: List[Tuple[Dict, Path]] = []
    up_to_date = 0
    for f in files:
        dest = cache_root / f["relpath"]
        expected_md5 = f.get("md5")
        if expected_md5:
            if _md5(dest) != expected_md5:
                to_download.append((f, dest))
            else:
                up_to_date += 1
        else:
            if not dest.exists():
                to_download.append((f, dest))
            else:
                up_to_date += 1

    print(f"Need to download {len(to_download)} files... (up-to-date: {up_to_date})")

    # 4) Parallel download
    n_ok = n_skip = n_fail = 0
    if to_download:
        with ThreadPoolExecutor(max_workers=max_workers) as pool, tqdm(total=len(to_download), desc="Syncing Google Drive") as pbar:
            futures = []
            for f, dest in to_download:
                futures.append(
                    pool.submit(
                        _download_with_retries,
                        creds,
                        f["id"],
                        dest,
                        f.get("md5"),
                        chunk_mb,
                    )
                )
            for fut in as_completed(futures):
                status, msg = fut.result()
                if status == "ok":
                    n_ok += 1
                elif status == "skipped":
                    n_skip += 1
                else:
                    n_fail += 1
                pbar.update(1)

    # 5) Summary
    print(f"Done. downloaded={n_ok}, skipped={n_skip}, failed={n_fail}, up_to_date={up_to_date}")
    if n_fail > 0:
        print("Some files failed to download. Re-run ensure_local_tree() to retry; transient errors usually clear on retry.")

    return str(cache_root.resolve())