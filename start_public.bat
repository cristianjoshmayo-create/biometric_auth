@echo off
cd /d "%~dp0"
echo Launching server (venv310) and ngrok tunnel in separate PowerShell windows...
echo.

start "Biometric Auth - Server" powershell -NoExit -Command "cd '%~dp0'; & .\venv310\Scripts\Activate.ps1; cd backend; uvicorn main:app --host 127.0.0.1 --port 8000"

start "Biometric Auth - ngrok" powershell -NoExit -Command "cd '%~dp0'; & .\venv310\Scripts\Activate.ps1; .\ngrok.exe http 8000"

echo Both PowerShell windows launched. Close them manually when done.
