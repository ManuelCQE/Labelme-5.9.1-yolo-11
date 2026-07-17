@echo off
setlocal
title LabelMe + YOLO11

set "ROOT=%~dp0"
set "ENV_DIR=%ROOT%miniconda\envs\labelme-env"
set "PYTHON_EXE=%ENV_DIR%\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERREUR] Installation non trouvée.
    echo Veuillez d'abord exécuter install.bat
    pause & exit /b 1
)

:: Lancer LabelMe
"%PYTHON_EXE%" -m labelme
