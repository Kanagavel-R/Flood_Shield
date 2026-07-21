# ==========================================
# Flood Shield
# Main Program
# ==========================================

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

print("=" * 60)
print("FLOOD SHIELD PROJECT")
print("=" * 60)

print("\nRunning Module 1...")
subprocess.run(
    [sys.executable, os.path.join(SRC_DIR, "preprocessing.py")],
    check=True
)

print("\nRunning Module 2...")
subprocess.run(
    [sys.executable, os.path.join(SRC_DIR, "supervised.py")],
    check=True
)

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)