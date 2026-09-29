"""Runtime artifacts live outside the published teaching directory."""
import os
import tempfile
from pathlib import Path
OUTPUT_DIR = Path(os.environ.get("COURSE_OUTPUT_DIR", str(Path(tempfile.gettempdir()) / ("tilelang_course_" + str(os.getuid())) / "04_gemm")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
