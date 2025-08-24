#!/usr/bin/env python3
"""
Windows Service Deployment Script for DocuChat Web UI

This script creates a Windows service that automatically starts DocuChat
web interface on system boot. Requires administrative privileges.

Usage:
    python deploy_windows_service.py install
    python deploy_windows_service.py start
    python deploy_windows_service.py stop
    python deploy_windows_service.py remove
"""

import sys
import os
import time
import subprocess
from pathlib import Path

try:
    import win32service
    import win32serviceutil
    import win32event
    import servicemanager
    WINDOWS_SERVICE_AVAILABLE = True
except ImportError:
    WINDOWS_SERVICE_AVAILABLE = False

class DocuChatWindowsService(win32serviceutil.ServiceFramework):
    """Windows service wrapper for DocuChat Web UI"""
    
    _svc_name_ = "DocuChatWebUI"
    _svc_display_name_ = "DocuChat Web UI Service"
    _svc_description_ = "DocuChat RAG Web Interface - Document Q&A System"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.running = True
        
        # Service configuration
        self.app_path = Path(__file__).parent.parent
        self.python_path = sys.executable
        self.app_script = self.app_path / "web" / "web_app.py"
        self.log_path = self.app_path / "logs" / "service.log"
        
        # Ensure logs directory exists
        self.log_path.parent.mkdir(exist_ok=True)
        
        # Service process
        self.process = None
        
    def SvcStop(self):
        """Stop the service"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.running = False
        
        # Terminate the web app process
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except Exception as e:
                servicemanager.LogErrorMsg(f"Error stopping process: {e}")
                
    def SvcDoRun(self):
        """Run the service"""
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        
        self.main()
        
    def main(self):
        """Main service loop"""
        while self.running:
            try:
                # Start the web application
                servicemanager.LogInfoMsg("Starting DocuChat Web UI...")
                
                self.process = subprocess.Popen([
                    self.python_path,
                    str(self.app_script)
                ], 
                cwd=str(self.app_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
                )
                
                servicemanager.LogInfoMsg("DocuChat Web UI started successfully")
                
                # Wait for stop signal or process termination
                while self.running and self.process.poll() is None:
                    if win32event.WaitForSingleObject(self.hWaitStop, 1000) == win32event.WAIT_OBJECT_0:
                        break
                        
                # If process died unexpectedly, restart it
                if self.running and self.process.poll() is not None:
                    servicemanager.LogErrorMsg("DocuChat Web UI process died, restarting...")
                    time.sleep(5)  # Wait before restart
                    
            except Exception as e:
                servicemanager.LogErrorMsg(f"Error in service main loop: {e}")
                time.sleep(10)  # Wait before retry

def create_service_installer():
    """Create Windows service installation script"""
    
    install_script = '''@echo off
REM DocuChat Web UI Service Installation Script
REM Run as Administrator

echo Installing DocuChat Web UI as Windows Service...

REM Install the service
python deploy_windows_service.py install

REM Configure service to start automatically
sc config DocuChatWebUI start= auto
sc config DocuChatWebUI depend= "ollama"

REM Start the service
python deploy_windows_service.py start

echo.
echo Service installed successfully!
echo - Service Name: DocuChatWebUI
echo - Access URL: http://localhost:7860
echo - Logs: logs/service.log
echo.
echo To manage the service:
echo   net start DocuChatWebUI
echo   net stop DocuChatWebUI
echo   python deploy_windows_service.py remove

pause
'''
    
    with open('install_service.bat', 'w') as f:
        f.write(install_script)
        
    print("✅ Created install_service.bat")
    print("   Run as Administrator to install Windows service")

def create_service_config():
    """Create service configuration files"""
    
    # Create service configuration
    config = {
        "service": {
            "name": "DocuChatWebUI",
            "display_name": "DocuChat Web UI Service",
            "description": "DocuChat RAG Web Interface - Document Q&A System",
            "auto_start": True,
            "restart_on_failure": True
        },
        "application": {
            "host": "0.0.0.0",
            "port": 7860,
            "debug": False,
            "log_level": "INFO"
        },
        "dependencies": {
            "ollama_url": "http://localhost:11434",
            "model": "gemma3:270m"
        }
    }
    
    import json
    with open('service_config.json', 'w') as f:
        json.dump(config, f, indent=2)
        
    print("✅ Created service_config.json")

def main():
    """Main deployment function"""
    
    if not WINDOWS_SERVICE_AVAILABLE:
        print("❌ Windows service modules not available")
        print("   Install with: pip install pywin32")
        return 1
        
    if len(sys.argv) == 1:
        print("DocuChat Windows Service Deployment")
        print("Usage:")
        print("  python deploy_windows_service.py install")
        print("  python deploy_windows_service.py start")
        print("  python deploy_windows_service.py stop")
        print("  python deploy_windows_service.py remove")
        print("  python deploy_windows_service.py setup")
        return 0
        
    if 'setup' in sys.argv:
        create_service_installer()
        create_service_config()
        print("\n🚀 Windows service deployment files created!")
        print("   1. Run install_service.bat as Administrator")
        print("   2. Service will start automatically on system boot")
        print("   3. Access at http://localhost:7860")
        return 0
    
    # Handle service operations
    win32serviceutil.HandleCommandLine(DocuChatWindowsService)

if __name__ == '__main__':
    sys.exit(main())