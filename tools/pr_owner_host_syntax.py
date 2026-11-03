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

from pr_host_build_contract import contract


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--target-build-log', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gate = contract(args.candidate.read_text(), args.source.read_text(),
                    args.target_build_log.read_text())
    gate.update(candidate_sha256=digest(args.candidate), source_sha256=digest(args.source),
                target_build_log_sha256=digest(args.target_build_log))
    (args.out / 'language-contract.json').write_text(json.dumps(gate, indent=2) + chr(10))
    if not gate['target_language_contract_pass']:
        print(json.dumps(dict(gate, decision='STOP_TARGET_LANGUAGE_MISMATCH', compiler_started=False),
                         indent=2))
        raise SystemExit(2)
    compiler = Path('/usr/bin/g++')
    cuda_include = Path('/root/.cache/okc-cuda130/nvidia/cu13/include')
    header = cuda_include / 'cuda.h'
    if digest(header) != '7a79d67ca888061b1dfbe4739b51ad64f85a2f3ad874947ce09b2c635b4c7072':
        raise ValueError('fixed secondary CUDA13 header mismatch')
    torch_root = Path(importlib.util.find_spec('torch').origin).parent
    includes = [cuda_include, torch_root / 'include',
                torch_root / 'include/torch/csrc/api/include',
                Path(sysconfig.get_paths()['include'])]
    command = [str(compiler), gate['target_default_standard'], *gate['extra_cflags'], '-fsyntax-only',
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
    result.update(language_contract=gate)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + chr(10))
    print(json.dumps(result, indent=2))
    raise SystemExit(run.returncode)


if __name__ == '__main__':
    main()
