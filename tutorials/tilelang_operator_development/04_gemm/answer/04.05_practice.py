from pathlib import Path
import subprocess,sys
src=Path(__file__).resolve().parents[1]/'src'
for step in range(1,6):
    subprocess.run([sys.executable,str(src/'profile_practice.py'),str(step),'--validate-only'],check=True)
