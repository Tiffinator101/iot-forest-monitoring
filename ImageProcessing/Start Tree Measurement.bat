@echo off
rem Double-click to start the Tree Measurement app. It opens in the browser.
rem Close this window to stop the app.
cd /d "%~dp0"
".venv\Scripts\python.exe" app.py
pause
