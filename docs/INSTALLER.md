# 📦 DocuChat Cross-Platform Installation Guide

Complete guide for deploying DocuChat on Windows, macOS, and Linux using the easiest methods available in 2025.

## 🎯 **Quick Decision Guide**

| **User Type** | **Recommended Method** | **Why** |
|---------------|----------------------|---------|
| **End Users** | PyInstaller Executable | Single file, no setup required |
| **Developers** | Docker | Consistent environment, easy updates |
| **Enterprises** | Platform Scripts | Service integration, full control |
| **Offline Use** | Portable Distribution | Works without internet after setup |

---

## 🚀 **Method 1: PyInstaller Single Executable (EASIEST)**

Creates a single executable file that runs anywhere without Python installation.

### **Windows**
```cmd
# Install PyInstaller
pip install pyinstaller

# Create Windows executable
pyinstaller --onefile --windowed --name "DocuChat" --add-data "src;src" --add-data "requirements.txt;." web/web_app.py

# Result: DocuChat.exe (50-80MB)
```

### **macOS**
```bash
# Install PyInstaller
pip install pyinstaller

# Create Mac app bundle
pyinstaller --onefile --windowed --name "DocuChat" --add-data "src:src" --add-data "requirements.txt:." web/web_app.py

# Result: DocuChat (40-70MB)
```

### **Linux**
```bash
# Install PyInstaller
pip install pyinstaller

# Create Linux executable
pyinstaller --onefile --windowed --name "DocuChat" --add-data "src:src" --add-data "requirements.txt:." web/web_app.py

# Result: DocuChat (40-70MB)
```

### **GUI Method (Ultra Easy)**
```bash
# Install auto-py-to-exe for visual interface
pip install auto-py-to-exe

# Launch GUI builder
auto-py-to-exe

# In the GUI:
# 1. Script Location: Select web/web_app.py
# 2. Onefile: One File
# 3. Console Window: Window Based (hide console)
# 4. Additional Files: Add folder "src" and file "requirements.txt"
# 5. Output Directory: Choose where to save
# 6. Click "CONVERT .PY TO .EXE"
```

### **Advanced PyInstaller Options**
```bash
# With icon and optimizations
pyinstaller --onefile --windowed --name "DocuChat" \
    --icon="icon.ico" \
    --add-data "src:src" \
    --add-data "requirements.txt:." \
    --exclude-module matplotlib \
    --exclude-module jupyter \
    --hidden-import torch \
    --hidden-import transformers \
    web/web_app.py

# For smaller file size (optional - requires UPX)
pip install upx-ucl  # or download UPX separately
pyinstaller --onefile --upx-dir=/path/to/upx web/web_app.py
```

---

## 🐳 **Method 2: Docker Deployment (Works Everywhere)**

You already have Docker setup - just use it!

### **Using Docker Compose (Recommended)**
```bash
# Build and run (you already have this)
docker-compose up -d

# Access at: http://localhost:7860

# Stop services
docker-compose down

# Update and rebuild
docker-compose up -d --build
```

### **Manual Docker Commands**
```bash
# Build the image
docker build -t docuchat-web .

# Run with volume mounts
docker run -d \
  --name docuchat-app \
  -p 7860:7860 \
  -v $(pwd)/documents:/app/documents:ro \
  -v $(pwd)/chroma_web:/app/chroma_web \
  -v $(pwd)/logs:/app/logs \
  docuchat-web

# Check logs
docker logs docuchat-app

# Stop container
docker stop docuchat-app
```

### **Docker for Different Platforms**
```bash
# Build multi-platform images
docker buildx build --platform linux/amd64,linux/arm64 -t docuchat-web .

# Run on different systems
# Windows with Docker Desktop:
docker run -d -p 7860:7860 -v %cd%/documents:/app/documents:ro docuchat-web

# macOS:
docker run -d -p 7860:7860 -v $(pwd)/documents:/app/documents:ro docuchat-web

# Linux:
docker run -d -p 7860:7860 -v ./documents:/app/documents:ro docuchat-web
```

---

## 📁 **Method 3: Portable Python Distribution**

Create a self-contained folder that runs anywhere with Python.

### **Create Portable Distribution**
```bash
# Create portable directory
mkdir DocuChat-Portable
cd DocuChat-Portable

# Install all dependencies locally
pip install --target . -r ../requirements.txt

# Copy application files
cp -r ../src .
cp ../web/web_app.py .
cp ../web/run_web_app.py .
cp ../requirements.txt .
cp ../docuchat.py .

# Copy deployment files
cp -r ../deployment/deploy_* .
cp -r ../deployment/docker-compose.yml .
cp -r ../deployment/Dockerfile .
```

### **Create Launch Scripts**

**Windows (launch.bat):**
```batch
@echo off
title DocuChat - Starting Application
echo 🚀 Starting DocuChat Web Application...
echo.
echo Make sure Ollama is running: ollama serve
echo Required model: ollama pull gemma3:270m
echo.
python web/web_app.py
if errorlevel 1 (
    echo.
    echo ❌ Error starting DocuChat
    echo Check that Python and dependencies are installed
    echo.
)
pause
```

**macOS/Linux (launch.sh):**
```bash
#!/bin/bash
echo "🚀 Starting DocuChat Web Application..."
echo ""
echo "Make sure Ollama is running: ollama serve"
echo "Required model: ollama pull gemma3:270m"
echo ""

# Check if Python is available
if command -v python3 &> /dev/null; then
    python3 web/web_app.py
elif command -v python &> /dev/null; then
    python web/web_app.py
else
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi
```

### **Create Installation Guide**
```bash
# Create README.txt for portable distribution
cat > DocuChat-Portable/README.txt << 'EOF'
DocuChat Portable Distribution
==============================

REQUIREMENTS:
- Python 3.8 or higher must be installed
- Ollama must be installed and running

QUICK START:
1. Install Ollama from: https://ollama.ai
2. Start Ollama: ollama serve (keep this running)
3. Install model: ollama pull gemma3:270m
4. Run DocuChat:
   - Windows: Double-click launch.bat
   - Mac/Linux: Run ./launch.sh in terminal
5. Open browser to: http://localhost:7860

FOLDER STRUCTURE:
- web_app.py - Main application
- src/ - Application source code
- All Python dependencies are included

TROUBLESHOOTING:
- If port 7860 is busy, kill the process or set GRADIO_SERVER_PORT=7861
- If Ollama connection fails, check: curl http://localhost:11434/api/tags
- For permission issues on Mac/Linux: chmod +x launch.sh

SUPPORT:
Check docs/DEPLOYMENT_GUIDE.md for detailed information.
EOF
```

---

## 🖥️ **Method 4: Platform-Specific Deployment**

Use your existing deployment scripts for server/service installations.

### **Linux Server Deployment**
```bash
# Automated installation (you already have this)
chmod +x deployment/deploy_linux.sh
./deployment/deploy_linux.sh install

# What it does:
# ✅ Installs system dependencies (Python, Nginx, etc.)
# ✅ Sets up Ollama with required model
# ✅ Creates dedicated user account
# ✅ Installs DocuChat as systemd service
# ✅ Configures Nginx reverse proxy
# ✅ Sets up firewall rules
# ✅ Auto-starts on system boot

# Service management
sudo systemctl start docuchat-web
sudo systemctl stop docuchat-web
sudo systemctl restart docuchat-web
sudo systemctl status docuchat-web

# View logs
sudo journalctl -u docuchat-web -f

# Uninstall completely
./deployment/deploy_linux.sh uninstall
```

### **Windows Service Installation**
```cmd
# Setup Windows service (you already have this)
python deployment/deploy_windows_service.py setup

# Install as Administrator
.\install_service.bat

# Service management
net start DocuChatWebUI
net stop DocuChatWebUI
net restart DocuChatWebUI

# Check service status
sc query DocuChatWebUI

# Remove service
python deployment/deploy_windows_service.py remove
```

### **macOS Installation** 
```bash
# Create macOS LaunchAgent (create this script)
cat > install_macos.sh << 'EOF'
#!/bin/bash
# DocuChat macOS Installation Script

APP_DIR="/Applications/DocuChat"
PLIST_FILE="$HOME/Library/LaunchAgents/com.docuchat.webui.plist"

echo "🍎 Installing DocuChat for macOS..."

# Create application directory
sudo mkdir -p "$APP_DIR"
sudo cp -r . "$APP_DIR/"
sudo chown -R $USER "$APP_DIR"

# Create LaunchAgent plist
cat > "$PLIST_FILE" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.docuchat.webui</string>
    <key>ProgramArguments</key>
    <array>
        <string>python3</string>
        <string>$APP_DIR/web/web_app.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$APP_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
PLIST

# Load the service
launchctl load "$PLIST_FILE"

echo "✅ DocuChat installed and started"
echo "📱 Access at: http://localhost:7860"
EOF

chmod +x install_macos.sh
```

---

## 🔧 **Build Automation Scripts**

### **build_executable.py** - Complete Build Automation
```python
#!/usr/bin/env python3
"""
Complete build automation for DocuChat executables
Supports Windows, macOS, and Linux
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import zipfile

def get_platform_info():
    """Get current platform information"""
    system = platform.system()
    machine = platform.machine()
    
    if system == "Windows":
        return "windows", ".exe"
    elif system == "Darwin":
        return "macos", ""
    elif system == "Linux":
        return "linux", ""
    else:
        return "unknown", ""

def clean_build_dirs():
    """Clean previous build directories"""
    dirs_to_clean = ["build", "dist", "__pycache__"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"🧹 Cleaned {dir_name}/")

def build_executable():
    """Build executable for current platform"""
    platform_name, exe_suffix = get_platform_info()
    
    print(f"🔨 Building DocuChat executable for {platform_name}...")
    
    # Clean previous builds
    clean_build_dirs()
    
    # Base PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name", "DocuChat",
        "--add-data", f"src{os.pathsep}src",
        "--add-data", f"requirements.txt{os.pathsep}.",
        "--clean",
        "--noconfirm",
        # Hidden imports for common issues
        "--hidden-import", "torch",
        "--hidden-import", "transformers", 
        "--hidden-import", "sentence_transformers",
        "--hidden-import", "chromadb",
        "--hidden-import", "gradio",
        # Exclude unnecessary modules to reduce size
        "--exclude-module", "matplotlib",
        "--exclude-module", "jupyter",
        "--exclude-module", "IPython",
        "--exclude-module", "notebook",
        "web/web_app.py"
    ]
    
    # Add platform-specific options
    if platform_name == "windows":
        # Add icon if available
        if os.path.exists("icon.ico"):
            cmd.extend(["--icon", "icon.ico"])
    elif platform_name == "macos":
        if os.path.exists("icon.icns"):
            cmd.extend(["--icon", "icon.icns"])
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        exe_name = f"DocuChat{exe_suffix}"
        exe_path = Path(f"dist/{exe_name}")
        
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"✅ Build successful!")
            print(f"📁 Executable: {exe_path}")
            print(f"📊 File size: {size_mb:.1f} MB")
            
            # Create distribution package
            create_distribution_package(exe_path, platform_name)
            
            return True
        else:
            print(f"❌ Executable not found at {exe_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def create_distribution_package(exe_path, platform_name):
    """Create distribution package with executable and documentation"""
    
    dist_dir = Path(f"dist/DocuChat-{platform_name}")
    dist_dir.mkdir(exist_ok=True)
    
    # Copy executable
    shutil.copy2(exe_path, dist_dir)
    
    # Copy documentation
    docs_to_copy = [
        "README.md",
        "docs/DEPLOYMENT_GUIDE.md",
        "requirements.txt"
    ]
    
    for doc in docs_to_copy:
        if os.path.exists(doc):
            shutil.copy2(doc, dist_dir)
    
    # Create installation instructions
    create_install_instructions(dist_dir, platform_name)
    
    # Create archive
    archive_name = f"DocuChat-{platform_name}"
    shutil.make_archive(f"dist/{archive_name}", 'zip', dist_dir)
    
    print(f"📦 Distribution package created: dist/{archive_name}.zip")

def create_install_instructions(dist_dir, platform_name):
    """Create platform-specific installation instructions"""
    
    instructions = f"""DocuChat Installation Instructions
{'=' * 40}

Platform: {platform_name.title()}

PREREQUISITES:
1. Install Ollama from: https://ollama.ai
2. Start Ollama service: ollama serve (keep running)
3. Install required model: ollama pull gemma3:270m

INSTALLATION:
"""

    if platform_name == "windows":
        instructions += """1. Extract this zip file to a folder (e.g., C:\\DocuChat\\)
2. Double-click DocuChat.exe to start
3. Open browser to: http://localhost:7860

TROUBLESHOOTING:
- If Windows Defender blocks the exe, click "More info" → "Run anyway"
- If port 7860 is busy, the app will try port 7861 automatically
"""
    
    elif platform_name == "macos":
        instructions += """1. Extract this zip file to Applications folder
2. Right-click DocuChat → Open (first time only, due to Gatekeeper)
3. Open browser to: http://localhost:7860

TROUBLESHOOTING:
- If "unidentified developer" error: System Preferences → Security → Allow DocuChat
- If permission denied: chmod +x DocuChat
"""
    
    elif platform_name == "linux":
        instructions += """1. Extract this zip file to /opt/docuchat/ or ~/DocuChat/
2. Make executable: chmod +x DocuChat
3. Run: ./DocuChat
4. Open browser to: http://localhost:7860

TROUBLESHOOTING:
- If missing dependencies: sudo apt install python3-tk (on Ubuntu/Debian)
- If permission denied: chmod +x DocuChat
"""

    instructions += """
USAGE:
1. Click "Process Documents" tab
2. Enter folder path containing your documents (PDF, DOCX, TXT)
3. Click "Start Processing" and wait for completion
4. Switch to "Chat" tab to ask questions about your documents

SUPPORT:
- Check the included documentation files
- Ensure Ollama is running: curl http://localhost:11434/api/tags
- Default models: gemma3:270m (recommended for speed)

VERSION: Built from DocuChat source code
"""

    with open(dist_dir / "INSTALL.txt", "w", encoding="utf-8") as f:
        f.write(instructions)

def main():
    """Main build function"""
    print("🚀 DocuChat Build System")
    print("=" * 30)
    
    # Check if PyInstaller is installed
    try:
        subprocess.run(["pyinstaller", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ PyInstaller not found. Install with: pip install pyinstaller")
        return 1
    
    # Check if we're in the right directory
    if not os.path.exists("web/web_app.py"):
        print("❌ web/web_app.py not found. Run this script from the DocuChat directory.")
        return 1
    
    # Build executable
    success = build_executable()
    
    if success:
        print("\n🎉 Build completed successfully!")
        print("📁 Check the dist/ folder for your executable and distribution package")
        return 0
    else:
        print("\n❌ Build failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### **create_portable.py** - Portable Distribution Builder
```python
#!/usr/bin/env python3
"""
Create portable DocuChat distribution with all dependencies
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import platform

def create_portable():
    """Create portable DocuChat distribution"""
    
    portable_dir = Path("DocuChat-Portable")
    
    print("📦 Creating portable DocuChat distribution...")
    
    # Clean existing directory
    if portable_dir.exists():
        shutil.rmtree(portable_dir)
        print("🧹 Cleaned existing portable directory")
    
    portable_dir.mkdir()
    
    # Install dependencies to portable directory
    print("📥 Installing Python dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--target", str(portable_dir), 
            "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    # Copy application files
    print("📋 Copying application files...")
    files_to_copy = [
        "web/web_app.py",
        "web/run_web_app.py", 
        "docuchat.py",
        "requirements.txt"
    ]
    
    dirs_to_copy = [
        "src"
    ]
    
    for file_name in files_to_copy:
        if os.path.exists(file_name):
            shutil.copy2(file_name, portable_dir)
    
    for dir_name in dirs_to_copy:
        if os.path.exists(dir_name):
            shutil.copytree(dir_name, portable_dir / dir_name)
    
    # Copy optional files
    optional_files = [
        "docs/DEPLOYMENT_GUIDE.md",
        "README.md", 
        "deployment/docker-compose.yml",
        "deployment/Dockerfile"
    ]
    
    for file_name in optional_files:
        if os.path.exists(file_name):
            shutil.copy2(file_name, portable_dir)
    
    # Create launch scripts
    print("🚀 Creating launcher scripts...")
    create_launch_scripts(portable_dir)
    
    # Create configuration files
    create_config_files(portable_dir)
    
    # Create documentation
    create_portable_documentation(portable_dir)
    
    # Calculate size
    total_size = get_dir_size(portable_dir)
    print(f"✅ Portable distribution created: {portable_dir}")
    print(f"📊 Total size: {total_size:.1f} MB")
    
    return True

def create_launch_scripts(portable_dir):
    """Create platform-specific launcher scripts"""
    
    # Windows launcher
    windows_launcher = """@echo off
title DocuChat - Portable Application
cls

echo ================================================
echo           DocuChat Portable Edition            
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found!
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM Display Python version
for /f "tokens=2" %%i in ('python --version') do echo ✅ Found Python %%i

echo.
echo 🔍 Checking Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama service not running!
    echo.
    echo Please start Ollama:
    echo 1. Install from: https://ollama.ai
    echo 2. Run: ollama serve
    echo 3. Install model: ollama pull gemma3:270m
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Ollama service is running
)

echo.
echo 🚀 Starting DocuChat...
echo 📱 Web interface will open at: http://localhost:7860
echo.
echo Press Ctrl+C to stop the application
echo ================================================

REM Set Python path to include portable directory
set PYTHONPATH=%~dp0;%PYTHONPATH%

REM Start the application
python "%~dp0web/web_app.py"

echo.
if errorlevel 1 (
    echo ❌ DocuChat encountered an error
    echo Check the error messages above
) else (
    echo ✅ DocuChat stopped normally
)

echo.
pause
"""

    # Unix launcher (macOS/Linux)
    unix_launcher = """#!/bin/bash

clear

echo "================================================"
echo "          DocuChat Portable Edition            "
echo "================================================"
echo ""

# Function to check command availability
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python availability
if command_exists python3; then
    PYTHON_CMD="python3"
    echo "✅ Found $(python3 --version)"
elif command_exists python; then
    PYTHON_CMD="python"
    echo "✅ Found $(python --version)"
else
    echo "❌ Python not found!"
    echo ""
    echo "Please install Python 3.8 or higher:"
    echo "• macOS: brew install python3 or download from python.org"
    echo "• Linux: sudo apt install python3 (Ubuntu/Debian) or equivalent"
    echo ""
    exit 1
fi

echo ""
echo "🔍 Checking Ollama service..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama service is running"
else
    echo "❌ Ollama service not running!"
    echo ""
    echo "Please start Ollama:"
    echo "1. Install from: https://ollama.ai"
    echo "2. Run: ollama serve"
    echo "3. Install model: ollama pull gemma3:270m"
    echo ""
    exit 1
fi

echo ""
echo "🚀 Starting DocuChat..."
echo "📱 Web interface will open at: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop the application"
echo "================================================"

# Set Python path to include portable directory
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

# Start the application
"$PYTHON_CMD" "$(dirname "$0")/web/web_app.py"

echo ""
if [ $? -eq 0 ]; then
    echo "✅ DocuChat stopped normally"
else
    echo "❌ DocuChat encountered an error"
    echo "Check the error messages above"
fi

echo ""
read -p "Press Enter to continue..."
"""

    # Write launcher scripts
    with open(portable_dir / "launch.bat", "w", encoding="utf-8") as f:
        f.write(windows_launcher)
    
    with open(portable_dir / "launch.sh", "w", encoding="utf-8") as f:
        f.write(unix_launcher)
    
    # Make Unix launcher executable
    os.chmod(portable_dir / "launch.sh", 0o755)

def create_config_files(portable_dir):
    """Create configuration files for the portable version"""
    
    # Create settings configuration
    settings_config = """{
    "default_settings": {
        "chunk_size": 1000,
        "top_k": 5,
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "gemma3:270m",
        "vector_db_path": "./chroma_web",
        "collection_name": "docuchat_portable_embeddings"
    },
    "portable_mode": true,
    "auto_open_browser": true,
    "server_port": 7860
}"""
    
    with open(portable_dir / "config.json", "w") as f:
        f.write(settings_config)

def create_portable_documentation(portable_dir):
    """Create comprehensive documentation for portable version"""
    
    readme_content = """DocuChat Portable Distribution
=============================

This is a self-contained portable version of DocuChat that includes all Python 
dependencies. You only need Python 3.8+ and Ollama installed on your system.

QUICK START:
-----------
1. Install Ollama: https://ollama.ai
2. Start Ollama: ollama serve (keep running in background)
3. Install AI model: ollama pull gemma3:270m
4. Run DocuChat:
   • Windows: Double-click launch.bat
   • Mac/Linux: Open terminal, run: ./launch.sh
5. Open browser to: http://localhost:7860

SYSTEM REQUIREMENTS:
-------------------
• Python 3.8 or higher
• Ollama (for AI language model)
• 4GB RAM minimum (8GB recommended)
• 2GB free disk space
• Internet connection (for initial setup only)

SUPPORTED DOCUMENT FORMATS:
---------------------------
• PDF files (.pdf)
• Microsoft Word documents (.docx)
• Plain text files (.txt)

HOW TO USE:
-----------
1. Launch the application using the appropriate launcher script
2. Go to "Process Documents" tab
3. Enter the path to a folder containing your documents
4. Click "Start Processing" and wait for completion
5. Switch to "Chat" tab to ask questions about your documents
6. Use the "Settings" tab to adjust processing parameters

FOLDER STRUCTURE:
----------------
• web/web_app.py - Main application file
• launch.bat / launch.sh - Launcher scripts  
• src/ - Application source code
• Python packages - All dependencies included
• config.json - Configuration file
• README.txt - This file

CONFIGURATION:
--------------
Edit config.json to change default settings:
• chunk_size: Size of text chunks for processing (100-2000)
• top_k: Number of relevant chunks to retrieve (1-10)  
• ollama_base_url: URL where Ollama is running
• ollama_model: AI model name to use

TROUBLESHOOTING:
---------------
Common Issues:

1. "Python not found" error:
   - Install Python from https://python.org
   - Make sure "Add Python to PATH" is checked during installation

2. "Ollama service not running":
   - Start Ollama: ollama serve
   - Check if running: curl http://localhost:11434/api/tags

3. "Port 7860 already in use":
   - Change port in config.json
   - Or kill the process using the port

4. Permission denied (Mac/Linux):
   - Make launcher executable: chmod +x launch.sh

5. Documents not processing:
   - Check folder path is correct
   - Ensure documents are in supported formats (PDF, DOCX, TXT)
   - Check file permissions

6. Chat responses are poor:
   - Try different AI models: ollama pull llama2 or ollama pull mistral
   - Adjust chunk_size and top_k in settings
   - Process more relevant documents

ADVANCED USAGE:
--------------
• Custom Models: Change ollama_model in config.json to use different AI models
• Multiple Collections: Change collection_name to separate different document sets  
• Performance: Adjust chunk_size based on your documents (smaller for precise answers, larger for context)

UPDATING:
---------
To update DocuChat:
1. Download new portable version
2. Copy your config.json to preserve settings
3. Copy your chroma_web/ folder to preserve processed documents

SUPPORT:
--------
• Check included documentation files
• Verify Ollama is running: curl http://localhost:11434/api/tags
• Test with sample documents first
• Check Python version: python --version (needs 3.8+)

VERSION INFORMATION:
-------------------
This portable version was built from DocuChat source code and includes
all necessary Python dependencies for offline operation.

For more information, visit the DocuChat project repository.
"""

    with open(portable_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(readme_content)

def get_dir_size(path):
    """Get directory size in MB"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except (OSError, IOError):
        pass
    return total / (1024 * 1024)

def main():
    """Main function"""
    print("🚀 DocuChat Portable Distribution Builder")
    print("=" * 45)
    
    # Check if we're in the right directory
    if not os.path.exists("web/web_app.py"):
        print("❌ web/web_app.py not found. Run this script from the DocuChat directory.")
        return 1
    
    # Check Python version
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return 1
    
    success = create_portable()
    
    if success:
        print("\n🎉 Portable distribution created successfully!")
        print("📁 Folder: DocuChat-Portable/")
        print("📋 See README.txt for usage instructions")
        return 0
    else:
        print("\n❌ Failed to create portable distribution!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## 🛠️ **Multi-Platform Build Scripts**

### **build_all.py** - Universal Build Script
```python
#!/usr/bin/env python3
"""
Universal build script for all platforms and methods
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_build_method(method_name, script_name):
    """Run a specific build method"""
    print(f"\n{'='*20} {method_name} {'='*20}")
    
    if not os.path.exists(script_name):
        print(f"❌ {script_name} not found, skipping {method_name}")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              check=True, capture_output=True, text=True)
        print(f"✅ {method_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {method_name} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def build_docker():
    """Build Docker image"""
    print(f"\n{'='*20} Docker Build {'='*20}")
    
    try:
        # Build Docker image
        subprocess.run(["docker", "build", "-t", "docuchat-web", "."], check=True)
        print("✅ Docker image built successfully")
        
        # Also build with docker-compose
        if os.path.exists("deployment/docker-compose.yml"):
            subprocess.run(["docker-compose", "build"], check=True, cwd="deployment")
            print("✅ Docker Compose build completed")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker build failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ Docker not found, skipping Docker build")
        return False

def main():
    """Main build orchestrator"""
    print("🚀 DocuChat Universal Build System")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print("=" * 50)
    
    build_results = {}
    
    # Build executable
    build_results["PyInstaller"] = run_build_method(
        "PyInstaller Executable", "build_executable.py"
    )
    
    # Create portable distribution  
    build_results["Portable"] = run_build_method(
        "Portable Distribution", "create_portable.py"
    )
    
    # Build Docker image
    build_results["Docker"] = build_docker()
    
    # Summary
    print(f"\n{'='*20} Build Summary {'='*20}")
    
    successful_builds = []
    failed_builds = []
    
    for method, success in build_results.items():
        if success:
            successful_builds.append(method)
            print(f"✅ {method}")
        else:
            failed_builds.append(method)
            print(f"❌ {method}")
    
    print(f"\n📊 Results: {len(successful_builds)} successful, {len(failed_builds)} failed")
    
    if successful_builds:
        print(f"\n🎉 Available distributions:")
        if "PyInstaller" in successful_builds:
            exe_suffix = ".exe" if platform.system() == "Windows" else ""
            print(f"   📱 Executable: dist/DocuChat{exe_suffix}")
        if "Portable" in successful_builds:
            print(f"   📁 Portable: DocuChat-Portable/")
        if "Docker" in successful_builds:
            print(f"   🐳 Docker: docker run -p 7860:7860 docuchat-web")
    
    return 0 if successful_builds else 1

if __name__ == "__main__":
    sys.exit(main())
```

### **Platform-Specific Batch Files**

**build_windows.bat:**
```batch
@echo off
title DocuChat Build System - Windows
cls

echo ================================================
echo         DocuChat Build System - Windows        
echo ================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Install build dependencies
echo 📦 Installing build dependencies...
pip install pyinstaller auto-py-to-exe >nul 2>&1
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed
echo.

REM Run universal build
echo 🚀 Starting build process...
python build_all.py

echo.
echo ================================================
echo                Build Complete                  
echo ================================================
echo.
echo Check the following for your distributions:
echo • dist\DocuChat.exe - Windows executable
echo • DocuChat-Portable\ - Portable version
echo • Docker image: docuchat-web
echo.
pause
```

**build_unix.sh:**
```bash
#!/bin/bash

clear

echo "================================================"
echo "       DocuChat Build System - Unix/Linux      "
echo "================================================"
echo ""

# Check Python
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
    echo "✅ Found $(python3 --version)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"  
    echo "✅ Found $(python --version)"
else
    echo "❌ Python not found! Please install Python 3.8+"
    exit 1
fi

echo ""

# Install build dependencies
echo "📦 Installing build dependencies..."
$PYTHON_CMD -m pip install pyinstaller >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"
echo ""

# Run universal build
echo "🚀 Starting build process..."
$PYTHON_CMD build_all.py

echo ""
echo "================================================"
echo "                Build Complete                  "
echo "================================================"
echo ""
echo "Check the following for your distributions:"
echo "• dist/DocuChat - Executable for your platform"
echo "• DocuChat-Portable/ - Portable version" 
echo "• Docker image: docuchat-web"
echo ""

read -p "Press Enter to continue..."
```

---

## 🔍 **Troubleshooting Guide**

### **Common Build Issues**

#### **PyInstaller Problems**
```bash
# Issue: "Module not found" during runtime
# Solution: Add hidden imports
pyinstaller --onefile --hidden-import=torch --hidden-import=transformers web/web_app.py

# Issue: Executable too large (>100MB)
# Solution: Exclude unnecessary modules
pyinstaller --onefile --exclude-module matplotlib --exclude-module jupyter web/web_app.py

# Issue: Slow startup time
# Solution: Use UPX compression
pip install upx-ucl
pyinstaller --onefile --upx-dir=/path/to/upx web/web_app.py

# Issue: Windows Defender blocks executable
# Solution: Code sign the executable or add exclusion
# For exclusion: Windows Security → Virus & threat protection → Exclusions
```

#### **Docker Issues**
```bash
# Issue: Build fails with dependency errors
# Solution: Update base image and rebuild
docker system prune -a  # Clean Docker cache
docker-compose build --no-cache

# Issue: Container won't start
# Solution: Check logs and port conflicts
docker logs docuchat-web
lsof -i :7860  # Check port usage

# Issue: Ollama connection from container
# Solution: Use host.docker.internal
docker run -e OLLAMA_BASE_URL=http://host.docker.internal:11434 docuchat-web
```

#### **Portable Distribution Issues**
```bash
# Issue: "No module named 'torch'" in portable version
# Solution: Ensure all dependencies are installed to target directory
pip install --target ./portable-dir -r requirements.txt --no-deps
pip install --target ./portable-dir torch --no-deps

# Issue: Path issues on different platforms
# Solution: Use relative paths and os.path.join()
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"  # Unix
set PYTHONPATH=%~dp0;%PYTHONPATH%  # Windows
```

### **Runtime Issues**

#### **Ollama Connection Problems**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Pull required model
ollama pull gemma3:270m

# Test model
ollama run gemma3:270m "Hello"

# Check Ollama logs
journalctl -u ollama  # Linux systemd
# or check Ollama process output
```

#### **Port Conflicts**
```bash
# Find what's using port 7860
# Windows:
netstat -ano | findstr 7860
# Mac/Linux:
lsof -i :7860

# Use different port
export GRADIO_SERVER_PORT=7861
# or set in code:
interface.launch(server_port=7861)
```

#### **Memory Issues**
```bash
# Monitor memory usage
# Windows: Task Manager → Performance → Memory
# Mac: Activity Monitor → Memory
# Linux: free -h

# Reduce memory usage:
# 1. Use smaller chunk sizes (500-800)
# 2. Reduce top_k values (3-5)
# 3. Process fewer documents at once
# 4. Use lighter Ollama models (gemma3:270m instead of llama2:7b)
```

### **Performance Optimization**

#### **Startup Time**
```bash
# Pre-compile Python files
python -m compileall src/

# Use faster models
ollama pull tinyllama  # Very fast but lower quality
ollama pull gemma3:270m  # Good balance
ollama pull gemma:2b  # Faster than larger models
```

#### **Processing Speed**
```bash
# Optimize chunk size for your documents
# Small docs (1-2 pages): 500-800 chunk size
# Medium docs (5-10 pages): 800-1200 chunk size  
# Large docs (>10 pages): 1200-2000 chunk size

# Use SSD storage for vector database
# Increase batch size for embedding generation
# Use GPU if available (requires CUDA setup)
```

---

## 📊 **Complete Comparison Matrix**

| Feature | PyInstaller | Docker | Portable | Platform Scripts |
|---------|------------|--------|----------|-----------------|
| **File Size** | 50-80MB | ~500MB | 200-300MB | N/A |
| **Setup Time** | 5 min | 2 min | 10 min | 15 min |
| **User Friendliness** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Portability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Dependencies** | None | Docker only | Python only | Full system |
| **Security** | Code bundled | Containerized | Source visible | Service mode |
| **Updates** | Rebuild needed | Pull new image | Replace files | In-place |
| **Debugging** | Difficult | Easy | Easy | Easy |
| **Performance** | Native speed | Near native | Native speed | Native speed |
| **Multi-platform** | Build per OS | Same image | Same dist | Per platform |

---

## 🎯 **Quick Decision Tree**

```
Are you distributing to end users?
├── YES → Use PyInstaller executable (easiest for users)
└── NO → Are you a developer?
    ├── YES → Use Docker (consistent environment)
    └── NO → Are you deploying on servers?
        ├── YES → Use platform scripts (service integration)
        └── NO → Use portable distribution (simple setup)
```

---

## 📝 **Distribution Checklist**

### **Before Building**
- [ ] Test application works locally with `python web/web_app.py`
- [ ] Verify all dependencies in requirements.txt
- [ ] Ensure Ollama integration works
- [ ] Test document processing with sample files
- [ ] Check chat functionality with Q&A

### **After Building**
- [ ] Test executable/distribution on clean machine
- [ ] Verify folder selection works
- [ ] Test document processing end-to-end
- [ ] Confirm chat responses are accurate
- [ ] Test settings persistence
- [ ] Check error handling and user feedback

### **For Distribution**
- [ ] Include installation instructions
- [ ] Add Ollama setup guide
- [ ] Provide sample documents for testing
- [ ] Include troubleshooting section
- [ ] Add contact/support information
- [ ] Test on target operating systems
- [ ] Verify antivirus software compatibility (for executables)

---

## 🎉 **Success! You're Ready to Deploy**

Choose your preferred method:

1. **🥇 For most users:** PyInstaller executable - single file, no setup
2. **🥈 For developers:** Docker - consistent, easy to update  
3. **🥉 For enterprises:** Platform scripts - full control, service integration
4. **🏅 For offline use:** Portable distribution - works anywhere with Python

Your DocuChat application is now ready for cross-platform deployment! 🚀