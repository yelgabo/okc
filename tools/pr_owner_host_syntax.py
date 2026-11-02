"""CPU-only syntax check of the exact PR owner host translation unit."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sysconfig
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    compiler = Path('/usr/bin/g++')
    cuda_include = Path('/root/.cache/okc-cuda130/nvidia/cu13/include')
    header = cuda_include / 'cuda.h'
    if digest(header) != '7a79d67ca888061b1dfbe4739b51ad64f85a2f3ad874947ce09b2c635b4c7072':
        raise ValueError('fixed secondary CUDA13 header mismatch')
    torch_root = Path(importlib.util.find_spec('torch').origin).parent
    includes = [cuda_include, torch_root / 'include',
                torch_root / 'include/torch/csrc/api/include',
                Path(sysconfig.get_paths()['include'])]
    command = [str(compiler), '-std=c++17', '-O2', '-pthread', '-fsyntax-only',
               '-DTORCH_EXTENSION_NAME=pr_owner_host_syntax']
    for include in includes:
        command += ['-I', str(include)]
    command.append(str(args.source.resolve()))
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    run = subprocess.run(command, text=True, capture_output=True, timeout=180)
    elapsed = time.monotonic() - started
    (args.out / 'stdout.txt').write_text(run.stdout)
    (args.out / 'stderr.txt').write_text(run.stderr)
    result = dict(command=command, returncode=run.returncode, elapsed_s=elapsed,
                  compiler=subprocess.check_output([str(compiler), '--version'], text=True),
                  source_sha256=digest(args.source), source_bytes=args.source.stat().st_size,
                  cuda_header_sha256=digest(header), torch_version=importlib.metadata.version('torch'),
                  torch_extension_header_sha256=digest(torch_root / 'include/torch/extension.h'),
                  host_syntax_pass=run.returncode == 0,
                  linked=False, imported=False, driver_loaded=False, cuda_compilation=False,
                  target_abi_proved=False, gpu_execution=False)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + chr(10))
    print(json.dumps(result, indent=2))
    raise SystemExit(run.returncode)


if __name__ == '__main__':
    main()
