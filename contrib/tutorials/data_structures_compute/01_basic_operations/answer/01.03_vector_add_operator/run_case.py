import subprocess
import sys


def build_command(executable):
    return [
        executable,
        "--length",
        "32768",
        "--block-dim",
        "4",
        "--tile-count",
        "16",
        "--seed",
        "7",
        "--warmup",
        "1",
        "--iterations",
        "5",
    ]


def metric_pass(text):
    required = [
        "length=32768",
        "block_dim=4",
        "tile_count=16",
        "tile_length=512",
        "correctness=PASS",
    ]
    return all(field in text for field in required)


result = subprocess.run(build_command(sys.argv[1]), text=True, capture_output=True)
print(result.stdout)
passed = result.returncode == 0 and metric_pass(result.stdout)
print("PRACTICE PASS" if passed else "PRACTICE TODO")
raise SystemExit(0 if passed else 1)
