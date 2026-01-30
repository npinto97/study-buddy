@echo off
echo Starting Study Buddy Analytics Dashboard (Local Only)...

if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found. Please run setup_env.bat first.
    pause
    exit /b
)

:: Run streamlit on a different port (e.g., 8502) to avoid conflict with the main app if running
"venv\Scripts\python.exe" -m streamlit run analysis_dashboard.py --server.port 8502

pause
