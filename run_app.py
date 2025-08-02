#!/usr/bin/env python3
"""
Launcher script for the Facial Landmarks Analysis Streamlit App
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages if not already installed."""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements. Please install manually:")
        print("pip install -r requirements.txt")
        return False
    return True

def run_streamlit_app():
    """Launch the Streamlit application."""
    print("🚀 Launching Facial Landmarks Analysis App...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

def main():
    """Main launcher function."""
    print("=" * 60)
    print("🔬 Facial Landmarks Analysis - Streamlit App Launcher")
    print("=" * 60)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    # Check if streamlit_app.py exists
    if not os.path.exists("streamlit_app.py"):
        print("❌ streamlit_app.py not found!")
        return
    
    # Install requirements
    if install_requirements():
        print("\n" + "=" * 60)
        run_streamlit_app()

if __name__ == "__main__":
    main()
