@echo off
setlocal
title Diagnostic Dubas - Palmier Dattier

echo ======================================================================
echo  Application de Classification des Feuilles de Palmier Dattier
echo  Detection du Dubas (Ommatissus lybicus)
echo ======================================================================
echo.

cd /d "%~dp0"

set "PY_EXE=C:\Users\malak\anaconda3\python.exe"

if exist "%PY_EXE%" (
    echo [OK] Environnement Python detecte : %PY_EXE%
    set "PATH=C:\Users\malak\anaconda3;C:\Users\malak\anaconda3\Scripts;%PATH%"
    "%PY_EXE%" -m streamlit run app_streamlit.py
    goto fin
)

python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Python detecte dans le PATH.
    python -m streamlit run app_streamlit.py
    goto fin
)

echo [ERREUR] Python est introuvable a l'emplacement C:\Users\malak\anaconda3\python.exe
pause

:fin
endlocal
