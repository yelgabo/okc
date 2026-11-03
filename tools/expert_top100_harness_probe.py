"""Read-only CPU prerequisite probe; imports no Torch/CUDA and installs nothing."""
import importlib.util
import json
import platform
import sys

if __name__ == '__main__':
    spec = importlib.util.find_spec('torch')
    print(json.dumps(dict(executable=sys.executable, version=sys.version, platform=platform.platform(),
        torch_spec=None if spec is None else spec.origin,
        status='STOP_CPU_HOST_GATE' if spec is None else 'TORCH_PRESENT_NOT_EXECUTED'), indent=2))
