#!/usr/bin/env python3
"""
DocuChat Web Application Launcher

Simple launcher script for the DocuChat web application with environment checks.
"""

import sys
import os
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    return True


def check_ollama():
    """Check if Ollama is available."""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Ollama found: {result.stdout.strip()}")
            return True
        else:
            print("⚠️  Ollama found but may not be working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama not found")
        print("   Install from: https://ollama.ai")
        return False


def check_ollama_service():
    """Check if Ollama service is running."""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama service running with {len(models)} models")
            
            # Check for recommended models
            model_names = [model.get('name', '') for model in models]
            recommended = ['gemma3:270m', 'llama3.1:8b', 'mistral:7b']
            
            found_models = [m for m in recommended if any(m in name for name in model_names)]
            if found_models:
                print(f"   Recommended models found: {', '.join(found_models)}")
            else:
                print("   No recommended models found. Consider:")
                print("   ollama pull gemma3:270m")
                
            return True
        else:
            print("⚠️  Ollama service responded with error")
            return False
            
    except Exception as e:
        print("❌ Ollama service not running")
        print("   Start with: ollama serve")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'gradio', 'torch', 'transformers', 'sentence_transformers', 
        'chromadb', 'numpy', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("   or")
        print("   pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main launcher function."""
    print("🚀 DocuChat Web Application Launcher")
    print("=" * 50)
    
    # Check Python version
    print("\n📋 Checking Python version...")
    if not check_python_version():
        return 1
    print(f"✅ Python {sys.version.split()[0]}")
    
    # Check dependencies
    print("\n📦 Checking Python dependencies...")
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them first.")
        return 1
    
    # Check Ollama
    print("\n🤖 Checking Ollama...")
    ollama_available = check_ollama()
    ollama_running = check_ollama_service() if ollama_available else False
    
    if not ollama_available:
        print("\n❌ Ollama is required but not found.")
        print("   1. Install from: https://ollama.ai")
        print("   2. Start service: ollama serve")
        print("   3. Pull a model: ollama pull gemma3:270m")
        return 1
    
    if not ollama_running:
        print("\n⚠️  Ollama is installed but service is not running.")
        print("   Start with: ollama serve")
        print("   Then run this launcher again.")
        return 1
    
    # All checks passed - launch web app
    print("\n🎉 All checks passed! Starting DocuChat Web Application...")
    print("\n" + "=" * 50)
    
    try:
        # Import and run the web app
        from web_app import main as web_app_main
        return web_app_main()
        
    except ImportError as e:
        print(f"❌ Failed to import web application: {e}")
        print("   Make sure web_app.py is in the same directory")
        return 1
    
    except Exception as e:
        print(f"❌ Failed to start web application: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())