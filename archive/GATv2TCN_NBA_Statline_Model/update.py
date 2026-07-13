"""
update.py — Update Script for the main model (PlayerAlpha)
============================================================
This script fetches the latest NBA box scores and rebuilds 
the necessary tensor artifacts for the model pipeline.

It explicitly runs:
1. scripts/01_fetch_data.py
2. scripts/02_build_tensors.py

If new players are found, it outputs a message indicating that
the model needs to be retrained.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    print("=======================================")
    print("1. Fetching Latest Data")
    print("=======================================")
    # Always resolve paths relative to this file so calls work
    # regardless of the current working directory.
    fetch_script = ROOT / "scripts" / "01_fetch_data.py"
    cmd_fetch = [sys.executable, str(fetch_script)]
    res_fetch = subprocess.run(cmd_fetch, cwd=str(ROOT))
    if res_fetch.returncode != 0:
        print("Data fetch failed!")
        sys.exit(res_fetch.returncode)

    print("\n=======================================")
    print("2. Building Tensors")
    print("=======================================")
    build_script = ROOT / "scripts" / "02_build_tensors.py"
    cmd_build = [sys.executable, str(build_script)]
    res_build = subprocess.run(cmd_build, cwd=str(ROOT), capture_output=True, text=True)
    
    # We print the output so the user can see it
    print(res_build.stdout)
    if res_build.returncode != 0:
        print(res_build.stderr)
        print("Tensor building failed!")
        sys.exit(res_build.returncode)
        
    print(res_build.stderr)

    if "RETRAIN REQUIRED" in res_build.stdout or "RETRAIN REQUIRED" in res_build.stderr:
        print("\n" + "*"*60)
        print("🚨 **RETRAIN REQUIRED** 🚨")
        print("New players were added to or removed from the dataset.")
        print("You MUST retrain the GATv2 backbone and the Kalshi calibrator.")
        print("Please run:")
        print("  1. python scripts/03_train.py")
        print("  2. python scripts/04_calibrate.py")
        print("  3. Check the playeralpha sub-model dataset as well.")
        print("*"*60 + "\n")
    else:
        print("\n✅ Data is fully up to date and model does NOT need to be retrained.")

if __name__ == "__main__":
    main()
