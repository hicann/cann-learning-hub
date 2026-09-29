"""Select the configured course fork when Python is launched via src/env.sh."""
import os
import sys
from pathlib import Path
root = os.environ.get("TILELANG_COURSE_FORK")
if root and (Path(root) / "tilelang/__init__.py").is_file():
    # An editable install may otherwise redirect tilelang to a different checkout.
    sys.meta_path = [f for f in sys.meta_path if type(f).__name__ != "ScikitBuildRedirectingFinder"]
    sys.path.insert(0, root)
