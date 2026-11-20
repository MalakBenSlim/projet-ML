@echo off
setlocal
title Diagnostic Dubas - Palmier Dattier

echo ======================================================================
echo  Application de Classification des Feuilles de Palmier Dattier
echo  Detection du Dubas (Ommatissus lybicus)
echo ======================================================================
echo.

cd /d "%~dp0"

set "PY_EXE="
if exist "C:\Espressif\tools\python\python.exe" set "PY_EXE=C:\Espressif\tools\python\python.exe"
if not defined PY_EXE if exist "C:\Users\CDA EXPERTISE\anaconda3\python.exe" set "PY_EXE=C:\Users\CDA EXPERTISE\anaconda3\python.exe"
if not defined PY_EXE if exist "C:\Users\malak\anaconda3\python.exe" set "PY_EXE=C:\Users\malak\anaconda3\python.exe"

if defined PY_EXE (
    echo [OK] Environnement Python detecte : %PY_EXE%
    "%PY_EXE%" -m streamlit run app_streamlit.py
    goto fin
)

python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Python detecte dans le PATH.
    python -m streamlit run app_streamlit.py
    goto fin
)

echo [ERREUR] Python est introuvable.
pause

:fin
endlocal
