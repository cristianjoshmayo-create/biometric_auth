@echo off
cd /d "%~dp0"
echo Starting Biometric Auth API...
echo.
echo [1] Testing / Enrollment  (no reload — stable, use this for demos and thesis)
echo [2] Development           (auto-reload on .py changes, do NOT use during enrollment)
echo.
set /p MODE="Choose mode (1 or 2): "

cd backend

if "%MODE%"=="2" (
    echo Starting in DEVELOPMENT mode (auto-reload ON)...
    uvicorn main:app ^
        --host 127.0.0.1 ^
        --port 8000 ^
        --reload ^
        --reload-dir . ^
        --reload-include "*.py" ^
        --reload-exclude "*.pkl" ^
        --reload-exclude "*.pth" ^
        --reload-exclude "*.pyc" ^
        --reload-exclude "*.db" ^
        --reload-exclude "__pycache__"
) else (
    echo Starting in TESTING mode (auto-reload OFF)...
    uvicorn main:app ^
        --host 127.0.0.1 ^
        --port 8000
)