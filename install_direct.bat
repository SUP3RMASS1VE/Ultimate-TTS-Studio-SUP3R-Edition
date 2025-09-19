@echo off
setlocal enabledelayedexpansion
echo.
echo ========================================
echo  Ultimate TTS Studio SUP3R Edition
echo     DIRECT INSTALLER
echo ========================================
echo.

REM Check if we're in a conda environment
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda is not activated!
    echo Please run this from an Anaconda/Miniconda prompt.
    pause
    exit /b 1
)

echo [INFO] Starting installation process...
echo.

REM Get current directory
set "APP_DIR=%cd%"
set "ENV_NAME=tts_env"
set "ENV_PATH=%APP_DIR%\%ENV_NAME%"

REM Step 1: Create or update conda environment in local directory
echo [STEP 1/6] Checking conda environment...
if exist "%ENV_PATH%" (
    echo [INFO] Environment already exists at "%ENV_PATH%"
    echo [INFO] Activating existing environment...
) else (
    echo [INFO] Creating new conda environment in "%ENV_PATH%"...
    echo [INFO] This may take a few minutes...
    call conda create --prefix "%ENV_PATH%" python=3.10 -y >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create conda environment!
        pause
        exit /b 1
    )
)
echo [SUCCESS] Conda environment ready!
echo.

REM Step 2: Activate the environment
echo [STEP 2/6] Activating conda environment...
call conda activate "%ENV_PATH%"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to activate conda environment!
    echo [INFO] Trying alternative activation method...
    call activate "%ENV_PATH%"
    if !errorlevel! neq 0 (
        echo [ERROR] Still failed to activate environment!
        pause
        exit /b 1
    )
)
echo [SUCCESS] Environment activated!
echo.

REM Verify we're in the right environment
echo [INFO] Current Python location:
where python
echo.

REM Step 3: Install UV using pip
echo [STEP 3/6] Installing UV package manager...
call python -m pip install --upgrade pip
call python -m pip install uv
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install UV!
    pause
    exit /b 1
)
echo [SUCCESS] UV installed successfully!
echo.

REM Step 4: Install requirements.txt using UV
echo [STEP 4/6] Installing requirements from requirements.txt...
if exist "%APP_DIR%\requirements.txt" (
    call python -m uv pip install -r "%APP_DIR%\requirements.txt"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install requirements!
        pause
        exit /b 1
    )
    echo [SUCCESS] Requirements installed successfully!
) else (
    echo [WARNING] requirements.txt not found, skipping...
)
echo.

REM Step 5: Install pynini using conda
echo [STEP 5/6] Installing pynini from conda-forge...
call conda install -c conda-forge pynini==2.1.6 -y
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install pynini!
    pause
    exit /b 1
)
echo [SUCCESS] pynini installed successfully!
echo.

REM Step 6: Install PyTorch GPU version (reinstall to ensure GPU support)
echo [STEP 6/10] Installing/Reinstalling PyTorch with GPU support...
echo [INFO] This will ensure PyTorch has CUDA support even if CPU version was installed via dependencies...
echo [INFO] Using --no-deps to avoid upgrading other packages and causing conflicts...
call python -m uv pip uninstall torch torchvision torchaudio
call python -m uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --no-deps
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install PyTorch GPU!
    echo [INFO] Falling back to CPU version...
    call python -m uv pip install torch torchvision torchaudio --no-deps
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install PyTorch CPU fallback!
        pause
        exit /b 1
    )
    echo [WARNING] PyTorch CPU version installed as fallback!
) else (
    echo [SUCCESS] PyTorch GPU version installed successfully!
)
echo.

REM Step 7: Install WeTextProcessing using UV
echo [STEP 7/10] Installing WeTextProcessing using UV...
call python -m uv pip install WeTextProcessing --no-deps
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install WeTextProcessing!
    pause
    exit /b 1
)
echo [SUCCESS] WeTextProcessing installed successfully!
echo.

REM Step 8: Install Triton for Windows
echo [STEP 8/10] Installing Triton for Windows...
echo [INFO] Installing Triton Windows version for enhanced GPU performance...
call python -m uv pip install triton-windows==3.3.1.post19
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install Triton Windows!
    echo [INFO] This may affect some model performance but installation can continue...
) else (
    echo [SUCCESS] Triton Windows installed successfully!
)
echo.

REM Step 9: Install Flash Attention for Windows
echo [STEP 9/10] Installing Flash Attention for Windows...
echo [INFO] Installing Flash Attention wheel for Windows with CUDA 12.8...
set "FLASH_ATTN_URL=https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl"
call python -m uv pip install "!FLASH_ATTN_URL!"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install Flash Attention!
    echo [INFO] This may affect some model performance but installation can continue...
) else (
    echo [SUCCESS] Flash Attention installed successfully!
)
echo.

REM Step 10: Install ONNX Runtime GPU version
echo [STEP 10/10] Installing ONNX Runtime GPU version...
echo [INFO] Ensuring ONNX Runtime has GPU support...
call python -m uv pip install --upgrade --force-reinstall --no-deps --no-cache-dir onnxruntime-gpu==1.22.0
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install ONNX Runtime GPU!
    echo [INFO] This may affect some model performance but installation can continue...
) else (
    echo [SUCCESS] ONNX Runtime GPU installed successfully!
)
echo.

echo ========================================
echo  Installation completed successfully!
echo ========================================
echo.
echo Environment location: "%ENV_PATH%"
echo To activate this environment in the future, use:
echo   conda activate "%ENV_PATH%"
echo.
echo Press any key to close this window...
pause >nul 
