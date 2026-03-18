@echo off
echo ===========================================
echo  Setting up Python GPU + Drive API environment
echo ===========================================

REM --- Create virtual environment ---
cd C:\Dev\gpu
python -m venv gpu_env

echo Activating virtual environment...
call C:\Dev\gpu\gpu_env\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing core AI libraries...
python -m pip install --no-cache-dir onnxruntime-directml==1.24.3
python -m pip install --no-cache-dir torch-directml torchvision pillow

echo Installing Google Drive API libraries...
python -m pip install --no-cache-dir google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client requests tqdm

echo Installing missing dependencies ONNX complained about...
python -m pip install --no-cache-dir ml_dtypes typing_extensions flatbuffers packaging

echo Creating data cache folder at C:\Data\project_cache
mkdir C:\Data\project_cache

echo ===========================================
echo  Environment setup complete!
echo  To activate environment later:
echo     C:\Dev\gpu\gpu_env\Scripts\activate
echo ===========================================
pause
