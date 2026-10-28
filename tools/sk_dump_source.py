"""Write the skinny kernel's device-only CUDA source (every config instantiated) to a file, for
tools/compile_check.sh.

    uv run --with torch python tools/sk_dump_source.py muse_bf16_proj/submission.py out.cu
"""

import importlib.util
import sys


def load(path):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    mod = load(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        f.write(mod.sk_source(device_only=True))
    print(f"{len(mod.SK_CONFIGS)} configs")
