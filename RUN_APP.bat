@echo off
echo.
echo ========================================
echo  Ultimate TTS Studio SUP3R Edition
echo          APP LAUNCHER
echo ========================================
echo.

REM Set local environment path
set LOCAL_ENV_PATH=%cd%\tts_env

REM Check if local environment exists
if not exist "%LOCAL_ENV_PATH%" (
    echo [ERROR] Local environment not found at: %LOCAL_ENV_PATH%
    echo Please run RUN_INSTALLER.bat first!
    pause
    exit /b 1
)

REM Try to find Anaconda/Miniconda installation
set CONDA_ROOT=
set FOUND_CONDA=0

REM Check common installation paths
if exist "%USERPROFILE%\Anaconda3" (
    set CONDA_ROOT=%USERPROFILE%\Anaconda3
    set FOUND_CONDA=1
) else if exist "%USERPROFILE%\anaconda3" (
    set CONDA_ROOT=%USERPROFILE%\anaconda3
    set FOUND_CONDA=1
) else if exist "%USERPROFILE%\Miniconda3" (
    set CONDA_ROOT=%USERPROFILE%\Miniconda3
    set FOUND_CONDA=1
) else if exist "%USERPROFILE%\miniconda3" (
    set CONDA_ROOT=%USERPROFILE%\miniconda3
    set FOUND_CONDA=1
) else if exist "%ProgramData%\Anaconda3" (
    set CONDA_ROOT=%ProgramData%\Anaconda3
    set FOUND_CONDA=1
) else if exist "%ProgramData%\Miniconda3" (
    set CONDA_ROOT=%ProgramData%\Miniconda3
    set FOUND_CONDA=1
) else if exist "C:\Anaconda3" (
    set CONDA_ROOT=C:\Anaconda3
    set FOUND_CONDA=1
) else if exist "C:\Miniconda3" (
    set CONDA_ROOT=C:\Miniconda3
    set FOUND_CONDA=1
)

if "%FOUND_CONDA%"=="0" (
    echo [ERROR] Could not find Anaconda or Miniconda installation!
    pause
    exit /b 1
)

echo [INFO] Found conda at: %CONDA_ROOT%
echo [INFO] Launching Ultimate TTS Studio...
echo.

REM Create a temporary launch script
echo @echo off > temp_launch.bat
echo echo. >> temp_launch.bat
echo echo [INFO] Activating environment... >> temp_launch.bat
echo call conda activate "%LOCAL_ENV_PATH%" >> temp_launch.bat
echo if errorlevel 1 ( >> temp_launch.bat
echo     echo [ERROR] Failed to activate environment! >> temp_launch.bat
echo     pause >> temp_launch.bat
echo     del "%%~f0" >> temp_launch.bat
echo     exit /b 1 >> temp_launch.bat
echo ) >> temp_launch.bat
echo echo [INFO] Starting Ultimate TTS Studio... >> temp_launch.bat
echo echo [INFO] The interface will load shortly at http://127.0.0.1:7860 >> temp_launch.bat
echo echo [INFO] Your browser will open automatically in a few seconds... >> temp_launch.bat
echo echo [INFO] Press Ctrl+C to stop the server >> temp_launch.bat
echo echo. >> temp_launch.bat
echo python launch.py >> temp_launch.bat
echo del "%%~f0" >> temp_launch.bat

REM Launch in conda prompt
start "Ultimate TTS Studio" /D "%cd%" "%windir%\System32\cmd.exe" /K ""%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%" && temp_launch.bat"

REM Wait a moment for the window to launch
timeout /t 2 /nobreak >nul

echo [INFO] Ultimate TTS Studio is starting in a new window...
echo [INFO] Waiting for the server to start before opening browser...
echo.

REM Wait for the server to start (checking if port 7860 is listening)
echo [INFO] Checking server status...
:wait_for_server
timeout /t 2 /nobreak >nul
netstat -an | findstr :7860 | findstr LISTENING >nul
if errorlevel 1 (
    echo [INFO] Server is still starting up...
    goto wait_for_server
)

echo [INFO] Server is ready! Opening browser...
start "" "http://127.0.0.1:7860"

echo.
echo [SUCCESS] Ultimate TTS Studio is now running!
echo [INFO] Browser should have opened automatically.
echo [INFO] If not, manually open: http://127.0.0.1:7860
echo.
echo [INFO] This window can be closed.
echo.
pause 
