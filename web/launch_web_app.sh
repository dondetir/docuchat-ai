#!/bin/bash

# DocuChat Web Application Launcher Script
# Simple bash script to launch DocuChat web interface

echo "🚀 DocuChat Web Application Launcher"
echo "===================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "   Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

# Check if we're in the right directory
if [ ! -f "web_app.py" ]; then
    echo "❌ web_app.py not found in current directory"
    echo "   Please run this script from the DocuChat directory"
    exit 1
fi

# Check if Ollama is running
echo "🤖 Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama service is running"
else
    echo "❌ Ollama service is not running"
    echo "   Please start Ollama with: ollama serve"
    echo "   Then run this script again"
    exit 1
fi

echo ""
echo "🎯 All checks passed!"
echo "📱 Starting DocuChat Web Application..."
echo "🌐 The web interface will open in your browser"
echo ""
echo "Press Ctrl+C to stop the application"
echo "===================================="
echo ""

# Launch the application
python3 run_web_app.py