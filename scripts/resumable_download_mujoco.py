import os
import sys
import time
import urllib.request
import subprocess

WHEEL_URL = "https://files.pythonhosted.org/packages/49/7d/53c7c251f28b49520cb3d052ce519c5c2d3a3d2e05244109724128f6f059/mujoco-3.11.0-cp314-cp314-win_amd64.whl"
FILENAME = "mujoco-3.11.0-cp314-cp314-win_amd64.whl"

def download_with_resume(url: str, filename: str, max_retries: int = 20):
    for attempt in range(1, max_retries + 1):
        try:
            existing_size = os.path.getsize(filename) if os.path.exists(filename) else 0
            headers = {"User-Agent": "Mozilla/5.0"}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
                
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_range = resp.headers.get("Content-Range")
                if content_range:
                    # Partial content (206)
                    total_size = int(content_range.split("/")[-1])
                    mode = "ab"
                else:
                    total_size = int(resp.headers.get("Content-Length", 0))
                    existing_size = 0
                    mode = "wb"
                    
                print(f"[Attempt {attempt}] Downloading from {existing_size / 1024 / 1024:.2f} MB / {total_size / 1024 / 1024:.2f} MB...")
                with open(filename, mode) as f:
                    downloaded = existing_size
                    while True:
                        chunk = resp.read(64 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = (downloaded * 100 // total_size) if total_size else 0
                        print(f"\rProgress: {downloaded / 1024 / 1024:.2f} / {total_size / 1024 / 1024:.2f} MB ({pct}%)", end="", flush=True)
                        
            final_size = os.path.getsize(filename)
            if total_size and final_size >= total_size:
                print(f"\nDownload complete! File size: {final_size} bytes")
                return True
        except Exception as e:
            print(f"\n[Attempt {attempt}] Connection interrupted ({e}). Retrying in 2s...")
            time.sleep(2)
            
    return False

if __name__ == "__main__":
    if download_with_resume(WHEEL_URL, FILENAME):
        print(f"Installing {FILENAME}...")
        res = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--user", FILENAME])
        if res.returncode == 0:
            print("MuJoCo installed successfully!")
        else:
            sys.exit(res.returncode)
    else:
        print("Failed to download after retries.")
        sys.exit(1)
