"""Environment inspection (spec 2.1): print Python/CUDA/GPU/package/disk facts."""

import json
import shutil
import sys

sys.path.insert(0, "src")
from jev_qwen.utils import env_fingerprint

info = env_fingerprint()
info["disk_free_gb"] = round(shutil.disk_usage(".").free / 1e9, 1)
print(json.dumps(info, indent=2))
