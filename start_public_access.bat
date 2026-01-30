@echo off
echo Starting Study Buddy Public Server...
echo.

:: Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found. Please run setup_env.bat first.
    pause
    exit /b
)

:: Run the python launcher directly using the venv python
"venv\Scripts\python.exe" run_public_server.py

pause
