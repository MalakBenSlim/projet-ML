@echo off
setlocal
title PalmVision - Diagnostic & Deep Learning (Application Web Vite.js)

echo ======================================================================
echo  PalmVision - Vision Artificielle & Deep Learning pour Palmier Dattier
echo  Detection du Dubas du Palmier Dattier (Ommatissus lybicus)
echo ======================================================================
echo.

cd /d "%~dp0"

set "PY_EXE="
if exist "C:\Espressif\tools\python\python.exe" set "PY_EXE=C:\Espressif\tools\python\python.exe"
if not defined PY_EXE if exist "C:\Users\CDA EXPERTISE\anaconda3\python.exe" set "PY_EXE=C:\Users\CDA EXPERTISE\anaconda3\python.exe"
if not defined PY_EXE if exist "C:\Users\malak\anaconda3\python.exe" set "PY_EXE=C:\Users\malak\anaconda3\python.exe"

if not defined PY_EXE (
    python --version >nul 2>&1
    if %ERRORLEVEL% EQU 0 set "PY_EXE=python"
)

if not defined PY_EXE (
    echo [ERREUR] Environnement Python introuvable.
    pause
    exit /b 1
)

echo [OK] Environnement Python detecte : %PY_EXE%
echo [INFO] Demarrage du serveur Web et de l'API (Port 8000)...
echo [INFO] Ouverture automatique dans votre navigateur : http://localhost:8000/home
echo.

start "" "http://localhost:8000/home"
"%PY_EXE%" -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload

endlocal
