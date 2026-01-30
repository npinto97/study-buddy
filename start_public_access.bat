@echo off
echo Starting Study Buddy Public Server...
echo.

:: Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate
) else (
    echo Virtual environment not found. Please run setup_env.bat first.
    pause
    exit /b
)

:: Run the python launcher
python run_public_server.py

pause
