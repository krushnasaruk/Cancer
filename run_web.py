"""
Sesame AI Digital Twin — Web Application Dashboard Launcher.

Launches the high-performance FastAPI + Three.js 3D WebGL Digital Twin
and automatically opens the default web browser.

Usage:
    python run_web.py
"""

import os
import sys
import webbrowser
import threading
import time
import uvicorn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def open_browser():
    time.sleep(1.2)
    url = "http://localhost:8000"
    print(f"Opening Sesame Digital Twin Web Application: {url}")
    webbrowser.open(url)


def main():
    print("=" * 65)
    print("   SESAME AI DIGITAL TWIN — MODERN WEB APPLICATION DASHBOARD")
    print("=" * 65)
    print("Starting FastAPI Backend + Three.js WebGL Real-Time Stream...")
    print("Dashboard URL: http://localhost:8000")
    print("=" * 65)
    
    # Open browser in separate background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Uvicorn Server
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
