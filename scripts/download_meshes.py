"""
Download official Sesame Robot 3D STL meshes from GitHub (dorianborian/sesame-robot).
"""

import os
import sys
import urllib.request
import json

REPO_API_URL = "https://api.github.com/repos/dorianborian/sesame-robot/contents/hardware/printing"
RAW_BASE_URL = "https://raw.githubusercontent.com/dorianborian/sesame-robot/main/hardware/printing"
TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulation/model/meshes"))

os.makedirs(TARGET_DIR, exist_ok=True)

def fetch_file_list():
    req = urllib.request.Request(
        REPO_API_URL,
        headers={"User-Agent": "SesameDigitalTwinDownloader/1.0"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            return [item["name"] for item in data if item["name"].endswith(".stl") or item["name"].endswith(".step")]
    except Exception as e:
        print(f"API query failed ({e}), falling back to known STL filename list...")
        return [
            "R1-v117.stl", "R2-v117.stl", "R3-v117.stl", "R4-v117.stl",
            "L1-v117.stl", "L2-v117.stl", "L3-v117.stl", "L4-v117.stl",
            "top.stl", "bottom.stl", "frame.stl", "cover.stl", "chassis.stl",
            "Top.stl", "Bottom.stl", "Frame.stl", "Body.stl"
        ]

def download_file(filename):
    url = f"{RAW_BASE_URL}/{filename}"
    out_path = os.path.join(TARGET_DIR, filename)
    print(f"Downloading: {url} -> {out_path}")
    try:
        urllib.request.urlretrieve(url, out_path)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            print(f"  [OK] Saved {filename} ({os.path.getsize(out_path):,} bytes)")
            return True
        else:
            if os.path.exists(out_path):
                os.remove(out_path)
            return False
    except Exception as e:
        print(f"  [SKIP] {filename} not found at direct URL ({e})")
        return False

def main():
    print("=" * 60)
    print("DOWNLOADING OFFICIAL SESAME 3D STL MESHES FROM GITHUB")
    print("=" * 60)
    files = fetch_file_list()
    print(f"Discovered potential mesh files: {files}")
    
    downloaded = 0
    for f in files:
        if download_file(f):
            downloaded += 1
            
    print("-" * 60)
    print(f"Downloaded {downloaded} STL mesh files to {TARGET_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
