"""Full Exp answer: replace vmax by vexp in the same read-compute-write kernel."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_relu_basic import run
if __name__=='__main__':run('exp')
