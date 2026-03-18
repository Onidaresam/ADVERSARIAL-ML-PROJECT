import os
import io
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# ------------------------------
# AUTHENTICATION
# ------------------------------
def _get_creds(client_secret_path: str, token_path: str) -> Credentials:
    """Return valid Google OAuth credentials."""
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
            creds = flow.run_local_server(port=0)

        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")

    return creds


# ------------------------------
# DRIVE CLIENT
# ------------------------------
def _build_drive(creds: Credentials):
    """Return authenticated Drive API service."""
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ------------------------------
# LIST CHILDREN IN A FOLDER
# ------------------------------
def _list_children(drive, folder_id: str) -> List[Dict]:
    """Return list of files/folders under given Drive folder ID."""
    query = f"'{folder_id}' in parents and trashed = false"
    fields = "files(id, name, mimeType, md5Checksum), nextPageToken"

    items = []
    page = None

    while True:
        resp = drive.files().list(
            q=query, fields=fields, pageToken=page
        ).execute()

        items.extend(resp.get("files", []))
        page = resp.get("nextPageToken")

        if not page:
            break

    return items


def _is_folder(item: Dict) -> bool:
    return item.get("mimeType") == "application/vnd.google-apps.folder"


# ------------------------------
# DOWNLOAD FILE
# ------------------------------
def _download_file(drive, file_id: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)

    request = drive.files().get_media(fileId=file_id)
    with io.FileIO(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


# ------------------------------
# MD5 CHECK
# ------------------------------
def _md5(path: Path) -> Optional[str]:
    if not path.exists():
        return None

    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------
# WALK DRIVE TREE RECURSIVELY
# ------------------------------
def _walk_drive(drive, root_id: str, rel: Path = Path(".")) -> List:
    """
    Recursively walk a Google Drive folder and return a flat list of all downloadable files.
    Skips Google Docs/Sheets/Slides (they cannot be downloaded with get_media()).
    """
    results = []
    stack = [(root_id, rel)]
    discovered = 0

    GOOGLE_DOC_MIMES = (
        "application/vnd.google-apps.document",
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.google-apps.presentation",
        "application/vnd.google-apps.form",
        "application/vnd.google-apps.drawing",
        "application/vnd.google-apps.script",
    )

    while stack:
        folder_id, relpath = stack.pop()
        children = _list_children(drive, folder_id)
        discovered += len(children)

        if discovered % 300 == 0:
            print(f"  Discovered {discovered} items...")

        for item in children:
            mime = item.get("mimeType", "")
            name = item.get("name", "")

            # Skip Google Docs types (not directly downloadable)
            if mime.startswith("application/vnd.google-apps.") and mime not in [
                "application/vnd.google-apps.folder"
            ]:
                print(f"  [SKIP] Google Doc/Sheet/Slide: {name} ({mime})")
                continue

            # Folder, recurse
            if mime == "application/vnd.google-apps.folder":
                stack.append((item["id"], relpath / name))
            else:
                # Real binary file → we can sync it
                results.append({
                    "id": item["id"],
                    "name": name,
                    "relpath": relpath / name,
                    "md5": item.get("md5Checksum"),
                    "mime": mime,
                })

    return results


# ------------------------------
# MAIN SYNC FUNCTION
# ------------------------------
def ensure_local_tree(
    folder_id: str = "1bg1W_xGKiP5z0zM1CBIo7u6KbOm-1U0O",
    cache_root: str = r"C:\Data\project_cache",
    client_secret_path: str = r"C:\Dev\gpu\client_secret.json",
    token_path: str = os.path.join(os.path.expanduser("~"), ".credentials", "gdrive_token.json")
) -> str:

    creds = _get_creds(client_secret_path, token_path)
    drive = _build_drive(creds)

    print(f"Listing Google Drive folder: {folder_id}")
    files = _walk_drive(drive, folder_id)

    cache_root = Path(cache_root)
    to_download = []

    for f in files:
        local_path = cache_root / f["relpath"]

        if f["md5"]:
            if _md5(local_path) != f["md5"]:
                to_download.append((f, local_path))
        else:
            if not local_path.exists():
                to_download.append((f, local_path))

    print(f"Need to download {len(to_download)} files...")

    for f, dest in tqdm(to_download, desc="Syncing Google Drive"):
        _download_file(drive, f["id"], dest)

        if f["md5"]:
            assert _md5(dest) == f["md5"], f"Checksum mismatch on {dest}"

    return str(cache_root)