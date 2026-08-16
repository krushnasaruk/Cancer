"""
Sesame AI Digital Twin — Control Center Launcher.

Usage:
    python run_gui.py
"""

import os
import sys

# Ensure workspace root is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gui.app import main

if __name__ == "__main__":
    main()
