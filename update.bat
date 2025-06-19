@echo off
setlocal enabledelayedexpansion
title Ultimate TTS Studio SUP3R Edition - Updater
color 0e

echo.
echo ========================================
echo  Ultimate TTS Studio SUP3R Edition
echo               UPDATER
echo ========================================
echo.

REM Check if git is available
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed or not in PATH!
    echo Please install Git from: https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

REM Get current directory
set "APP_DIR=%cd%"
set "ENV_NAME=tts_env"
set "ENV_PATH=%APP_DIR%\%ENV_NAME%"

REM Check if we're in a git repository
if not exist ".git" (
    echo [INFO] Setting up Git repository for updates...
    echo [WARNING] This will replace all files with the latest version from GitHub.
    echo.
    
    REM Initialize git repository
    git init
    git remote add origin https://github.com/SUP3RMASS1VE/Ultimate-TTS-Studio-SUP3R-Edition.git
    
    echo [INFO] Downloading latest files...
    git fetch origin
    
    REM Create main branch and force reset to match origin
    echo [INFO] Setting up main branch...
    git checkout -b main
    git reset --hard origin/main
    
    echo [SUCCESS] Files updated successfully!
) else (
    echo [INFO] Updating files...
    
    REM Handle any local changes by stashing them
    git status --porcelain > temp_status.txt
    set /p git_status=<temp_status.txt
    del temp_status.txt
    
    if not "%git_status%"=="" (
        echo [INFO] Backing up local changes...
        git stash push -m "Auto-backup before update"
    )
    
    REM Fetch latest updates first
    echo [INFO] Fetching latest updates...
    git fetch origin
    
    REM Get current branch
    for /f "tokens=*" %%i in ('git branch --show-current') do set current_branch=%%i
    
    REM If no current branch, switch to main branch
    if "%current_branch%"=="" (
        echo [INFO] Switching to main branch...
        git checkout main
        if errorlevel 1 (
            echo [INFO] Creating main branch...
            git checkout -b main origin/main
            if errorlevel 1 (
                echo [ERROR] Failed to setup main branch!
                pause
                exit /b 1
            )
        )
    )
    
    REM Update to latest version
    echo [INFO] Updating to latest version...
    git reset --hard origin/main
    if errorlevel 1 (
        echo [ERROR] Failed to update files!
        pause
        exit /b 1
    )
    
    echo [SUCCESS] Files updated successfully!
)

REM Check if requirements.txt was updated
git diff HEAD~1 HEAD --name-only 2>nul | findstr "requirements.txt" >nul
if not errorlevel 1 (
    set "deps_changed=1"
) else (
    set "deps_changed=0"
)

REM Ask if user wants to update dependencies
echo.
echo ========================================
echo  Dependency Update Options
echo ========================================
if "%deps_changed%"=="1" (
    echo [INFO] requirements.txt has been updated!
    echo [RECOMMENDED] You should update your dependencies.
) else (
    echo [INFO] No changes detected in requirements.txt
    echo You can still update dependencies if needed.
)
echo.
echo Would you like to update Python packages?
echo [y] Yes - Update all dependencies
echo [n] No - Skip dependency update
echo.
set /p "update_deps=Enter your choice (y/n): "

if /i "%update_deps%"=="y" (
    echo.
    echo [INFO] Updating dependencies...
    echo.
    
    REM Check if conda environment exists
    if not exist "%ENV_PATH%" (
        echo [ERROR] Conda environment not found at: "%ENV_PATH%"
        echo Please run RUN_INSTALLER.bat first!
        echo.
        pause
        exit /b 1
    )
    
    REM Check if conda is available
    where conda >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Conda is not available!
        echo Please run this from an Anaconda/Miniconda prompt.
        pause
        exit /b 1
    )
    
    REM Activate the environment
    echo [INFO] Activating conda environment...
    call conda activate "%ENV_PATH%"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to activate conda environment!
        pause
        exit /b 1
    )
    
    REM Update pip and UV first
    echo [INFO] Updating pip and UV...
    call python -m pip install --upgrade pip
    call python -m pip install --upgrade uv
    
    REM Update requirements using UV
    echo [INFO] Updating requirements with UV...
    if exist "%APP_DIR%\requirements.txt" (
        call python -m uv pip install -r "%APP_DIR%\requirements.txt" --upgrade
        if !errorlevel! neq 0 (
            echo [ERROR] Failed to update requirements!
            echo [INFO] Trying without --upgrade flag...
            call python -m uv pip install -r "%APP_DIR%\requirements.txt"
        )
    )
    
    REM Update conda packages
    echo [INFO] Updating conda packages (pynini)...
    call conda update -c conda-forge pynini -y
    
    REM Reinstall WeTextProcessing to ensure compatibility
    echo [INFO] Reinstalling WeTextProcessing...
    call python -m uv pip install WeTextProcessing --no-deps --force-reinstall
    
    echo.
    echo [SUCCESS] All dependencies updated successfully!
)

echo.
echo ========================================
echo Update completed!
echo.
if /i "%update_deps%"=="y" (
    echo Environment is ready at: "%ENV_PATH%"
    echo To run the app, use: RUN_APP.bat
) else (
    echo You can run the app with: RUN_APP.bat
    echo To update dependencies later, run this updater again.
)
echo ========================================
echo.
pause 