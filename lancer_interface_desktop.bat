@echo off
setlocal
title Diagnostic Dubas - Palmier Dattier

echo ======================================================================
echo  Application Desktop de Classification des Feuilles de Palmier Dattier
echo ======================================================================
echo.

cd /d "%~dp0"

set "PY_EXE=C:\Users\malak\anaconda3\python.exe"

if exist "%PY_EXE%" (
    echo [OK] Environnement Python detecte : %PY_EXE%
    set "PATH=C:\Users\malak\anaconda3;C:\Users\malak\anaconda3\Scripts;%PATH%"
    "%PY_EXE%" app_desktop.py
    goto fin
)

python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python app_desktop.py
    goto fin
)

echo [ERREUR] Python introuvable.
pause

:fin
endlocal
