"""
Download all official Sesame Robot STL meshes from GitHub.
"""

import os
import sys
import urllib.request

FILES_TO_DOWNLOAD = [
    ("hardware/printing/stl/Bottom-Cover-v121.stl", "bottom_cover.stl"),
    ("hardware/printing/stl/Internal-Frame-v121.stl", "internal_frame.stl"),
    ("hardware/printing/stl/top-covers/Top-Cover-No-Ears-v100.stl", "top_cover.stl"),
    ("hardware/printing/stl/L1-v117.stl", "fl_hip.stl"),
    ("hardware/printing/stl/L2-v117.stl", "rl_hip.stl"),
    ("hardware/printing/stl/L3-v117.stl", "fl_knee.stl"),
    ("hardware/printing/stl/L4-v117.stl", "rl_knee.stl"),
    ("hardware/printing/stl/R1-v117.stl", "fr_hip.stl"),
    ("hardware/printing/stl/R2-v117.stl", "rr_hip.stl"),
    ("hardware/printing/stl/R3-v117.stl", "fr_knee.stl"),
    ("hardware/printing/stl/R4-v117.stl", "rr_knee.stl"),
]

TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulation/model/meshes"))
os.makedirs(TARGET_DIR, exist_ok=True)
RAW_BASE_URL = "https://raw.githubusercontent.com/dorianborian/sesame-robot/main"

def main():
    print("=" * 60)
    print("DOWNLOADING OFFICIAL SESAME 3D STL MESHES")
    print("=" * 60)
    
    success_count = 0
    for repo_path, local_name in FILES_TO_DOWNLOAD:
        url = f"{RAW_BASE_URL}/{repo_path}"
        out_path = os.path.join(TARGET_DIR, local_name)
        print(f"Fetching {local_name} from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
                content = resp.read()
                f.write(content)
            size_kb = len(content) / 1024
            print(f"  [OK] Saved {local_name} ({size_kb:.1f} KB)")
            success_count += 1
        except Exception as e:
            print(f"  [FAIL] Failed to download {local_name}: {e}")
            
    print("-" * 60)
    print(f"Successfully downloaded {success_count}/{len(FILES_TO_DOWNLOAD)} STL files.")
    print("=" * 60)

if __name__ == "__main__":
    main()
