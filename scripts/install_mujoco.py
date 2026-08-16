import json
import urllib.request
import subprocess
import sys

def main():
    print("Fetching PyPI releases for mujoco...")
    req = urllib.request.urlopen("https://pypi.org/pypi/mujoco/json", timeout=60)
    data = json.loads(req.read().decode())
    latest_version = data["info"]["version"]
    print(f"Latest version: {latest_version}")
    
    releases = data["releases"][latest_version]
    win_wheels = [r for r in releases if "win_amd64" in r["filename"]]
    print("Available Windows wheels:")
    for w in win_wheels:
        print(f" - {w['filename']} ({w['size']} bytes)")
        
    # Match standard cp314
    target = None
    for w in win_wheels:
        if "cp314-cp314-" in w["filename"]:
            target = w
            break
            
    if not target:
        # Fallback to any cp314
        for w in win_wheels:
            if "cp314" in w["filename"]:
                target = w
                break
                
    if not target:
        print("No cp314 wheel found!")
        return 1
        
    filename = target["filename"]
    url = target["url"]
    print(f"\nDownloading {filename} from {url}...")
    urllib.request.urlretrieve(url, filename)
    print("Downloaded successfully! Installing with pip...")
    res = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--user", filename])
    print(f"Pip install result: {res.returncode}")
    return res.returncode

if __name__ == "__main__":
    sys.exit(main())
