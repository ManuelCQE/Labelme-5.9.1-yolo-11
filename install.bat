@echo off
setlocal enabledelayedexpansion
title Installation LabelMe + YOLO11

echo ============================================================
echo  Installation LabelMe 5.9.1-Yolo-DWPose
echo ============================================================
echo.

set "ROOT=%~dp0"
set "CONDA_DIR=%ROOT%miniconda"
set "ENV_DIR=%CONDA_DIR%\envs\labelme-env"
set "PYTHON_EXE=%ENV_DIR%\python.exe"
set "PIP_EXE=%ENV_DIR%\Scripts\pip.exe"
set "CONDA_EXE=%CONDA_DIR%\Scripts\conda.exe"

if exist "%PYTHON_EXE%" (
    echo [OK] Environnement deja installe. Mise a jour des dependances...
    goto :install_deps
)

echo [1/5] Telechargement de Miniconda...
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
set "MINICONDA_INSTALLER=%ROOT%miniconda_installer.exe"

if not exist "%MINICONDA_INSTALLER%" (
    curl.exe -L --progress-bar "%MINICONDA_URL%" -o "%MINICONDA_INSTALLER%"
    if errorlevel 1 (
        echo [ERREUR] Impossible de telecharger Miniconda.
        pause & exit /b 1
    )
)
echo [OK] Miniconda telecharge.

echo.
echo [2/5] Installation de Miniconda dans %CONDA_DIR%...
"%MINICONDA_INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_DIR%
if errorlevel 1 (
    echo [ERREUR] Installation Miniconda echouee.
    pause & exit /b 1
)
echo [OK] Miniconda installe.
del "%MINICONDA_INSTALLER%"

echo.
echo [2b] Acceptation des conditions d'utilisation Anaconda...
"%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
"%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
echo [OK] Conditions acceptees.

echo.
echo [3/5] Creation de l'environnement labelme-env (Python 3.10 + pip)...
"%CONDA_EXE%" create -y -p "%ENV_DIR%" python=3.10 pip
if errorlevel 1 (
    echo [ERREUR] Creation de l'environnement echouee.
    pause & exit /b 1
)
echo [OK] Environnement cree.

:install_deps
echo.
echo [4/5] Detection GPU et installation PyTorch...

:: Détecter nvidia-smi
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_LABEL=CPU (pas de GPU NVIDIA detecte)"

nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo      Pas de GPU NVIDIA detecte — installation PyTorch CPU.
    echo      Note : l'inference sera tres lente sans GPU.
    goto :install_torch
)

:: Lire la version du driver NVIDIA
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do set "DRIVER_VER=%%i"
echo      Driver NVIDIA detecte : %DRIVER_VER%

:: Extraire la partie majeure du driver (avant le premier point)
for /f "tokens=1 delims=." %%a in ("%DRIVER_VER%") do set "DRIVER_MAJOR=%%a"

:: Choisir la version CUDA selon le driver
if %DRIVER_MAJOR% LSS 452 (
    echo      Driver trop ancien ^(^< 452^) — installation PyTorch CPU.
    echo      Mettez a jour vos drivers NVIDIA pour profiter du GPU.
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    set "TORCH_LABEL=CPU (driver trop ancien)"
) else if %DRIVER_MAJOR% LSS 526 (
    echo      Driver compatible CUDA 11.8 — installation PyTorch cu118...
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu118"
    set "TORCH_LABEL=CUDA 11.8"
) else if %DRIVER_MAJOR% LSS 536 (
    echo      Driver compatible CUDA 12.1 — installation PyTorch cu121...
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
    set "TORCH_LABEL=CUDA 12.1"
) else (
    echo      Driver compatible CUDA 12.6 — installation PyTorch cu126...
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
    set "TORCH_LABEL=CUDA 12.6"
)

:install_torch
echo      PyTorch : %TORCH_LABEL%
"%PIP_EXE%" install torch==2.9.0 torchvision==0.24.0 --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo [ERREUR] Installation PyTorch echouee.
    pause & exit /b 1
)

"%PIP_EXE%" install -r "%ROOT%requirements.txt"
if errorlevel 1 (
    echo [ERREUR] Installation des dependances echouee.
    pause & exit /b 1
)

cd /d "%ROOT%"
"%PIP_EXE%" install -e .
if errorlevel 1 (
    echo [ERREUR] Installation de labelme echouee.
    pause & exit /b 1
)
echo [OK] Dependances installees.

echo.
echo [5/5] Telechargement des modeles YOLO11 + DWPose...
"%PYTHON_EXE%" "%ROOT%download_models.py"
if errorlevel 1 (
    echo [ERREUR] Telechargement des modeles echoue.
    pause & exit /b 1
)

echo.
echo ============================================================
echo  Installation terminee ! [%TORCH_LABEL%]
echo  Double-cliquez sur launch.bat pour demarrer LabelMe.
echo ============================================================
echo.
pause
