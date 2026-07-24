import os
import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
app_path = PROJECT_DIR / "dashboard" / "app.py"

if __name__ == "__main__":
    print(f"Launching Streamlit MRI Preprocessing Dashboard from: {app_path}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(cmd)
