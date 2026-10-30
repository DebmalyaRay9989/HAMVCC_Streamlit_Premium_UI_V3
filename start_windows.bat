@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Install Python 3.12 or create .venv manually with Python 3.11 or newer.
    pause
    exit /b 1
  )
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Dependency installation failed. Check your connection and retry pip install -r requirements.txt.
    pause
    exit /b 1
  )
)
echo Open http://127.0.0.1:8050 in your browser. Press Ctrl+C to stop.
.venv\Scripts\python.exe app.py
pause
