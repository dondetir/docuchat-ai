@echo off
REM DocuChat Web Application Launcher Script for Windows
REM Simple batch script to launch DocuChat web interface

echo 🚀 DocuChat Web Application Launcher
echo ====================================

REM Check if Python is available
python --version > nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo    Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "web_app.py" (
    echo ❌ web_app.py not found in current directory
    echo    Please run this script from the DocuChat directory
    pause
    exit /b 1
)

REM Check if Ollama is running (Windows doesn't have curl by default, so we'll skip this check)
echo 🤖 Checking dependencies...
echo ✅ Python found

echo.
echo 🎯 Starting DocuChat Web Application...
echo 📱 The web interface will open in your browser
echo.
echo Press Ctrl+C to stop the application
echo ====================================
echo.

REM Launch the application
python run_web_app.py

pause