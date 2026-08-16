import sys
import os
import json
import urllib.request
import subprocess

def download_and_install():
    print("Querying PyPI for MuJoCo wheels...")
    req = urllib.request.urlopen("https://pypi.org/pypi/mujoco/json", timeout=30)
    data = json.loads(req.read().decode())
    releases = data["releases"]["3.11.0"]
    
    wheel_info = None
    for r in releases:
        if r["filename"] == "mujoco-3.11.0-cp314-cp314-win_amd64.whl":
            wheel_info = r
            break
            
    if not wheel_info:
        print("Could not find standard cp314-win_amd64 wheel!")
        return False
        
    filename = wheel_info["filename"]
    url = wheel_info["url"]
    total_size = wheel_info["size"]
    print(f"Downloading {filename} ({total_size / 1024 / 1024:.2f} MB)...")
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(filename, "wb") as f:
        downloaded = 0
        chunk_size = 64 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            done_mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            print(f"\rProgress: {done_mb:.2f} / {total_mb:.2f} MB ({downloaded * 100 // total_size}%)", end="", flush=True)
            
    print(f"\nDownload complete! Installing {filename}...")
    res = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", filename])
    print(f"Pip exit code: {res.returncode}")
    return res.returncode == 0

if __name__ == "__main__":
    download_and_install()
