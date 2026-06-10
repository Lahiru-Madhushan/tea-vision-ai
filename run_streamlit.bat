@echo off
cd /d "%~dp0"

set "VENV_PY=env\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo.
    echo ERROR: Project virtual environment not found.
    echo Create it with:  python -m venv env
    echo Then install deps: env\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo Tea Vision AI - starting with project virtual environment
echo Python: %CD%\%VENV_PY%
echo.

"%VENV_PY%" -m streamlit run ai-service\streamlit_app.py

pause
