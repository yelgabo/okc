#!/usr/bin/env python3
"""SASS and resource usage of a layer file's slice cgemv (CG_CUDA_SRC after its import-time edits), device-only build.

python tools/ly_cg_sass.py <layer file.py> [--out DIR] [--nvcc /usr/local/cuda/bin/nvcc]
Extracts CG_CUDA_SRC, applies _PDL8_CG_EDITS (and _PMT_CG_EDITS when present) as the file does at import, compiles
it with -DGEMV_DEVICE_ONLY to a cubin for sm_90a and prints `cuobjdump -res-usage` plus the opcode histogram of
cgemv40_kernel<128, 2>; the full SASS goes to DIR/<stem>.sass. No GPU needed.
"""
import collections
import pathlib
import re
import subprocess
import sys


def main():
    path = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else pathlib.Path(".build/cgsass")
    nvcc = sys.argv[sys.argv.index("--nvcc") + 1] if "--nvcc" in sys.argv else "nvcc"
    out.mkdir(parents=True, exist_ok=True)
    text = path.read_text()
    i = text.index('CG_CUDA_SRC = r"""\n') + len('CG_CUDA_SRC = r"""\n')
    src = text[i:text.index('\n"""\n', i) + 1]
    ns = {}
    for name in ("_PDL8_CG_EDITS", "_PMT_CG_EDITS"):
        j = text.find(name + " = [")
        if j < 0:
            continue
        exec(text[j:text.index("\n", j)], ns)  # noqa: S102  (one-line list literals)
        for old, new, n in ns[name]:
            assert src.count(old) == n, (name, old[:60])
            src = src.replace(old, new)
    # the host code that instantiates the kernels is compiled out, so instantiate the slice's kernel explicitly
    src += ("\nnamespace { template __global__ void cgemv40_kernel<128, 2, false>(const uint8_t*, int, const int2*, int, unsigned, "
            "const __nv_bfloat16*, __nv_bfloat16*); }\n") if "bool TL" in src else \
           ("\nnamespace { template __global__ void cgemv40_kernel<128, 2>(const uint8_t*, int, const int2*, int, unsigned, "
            "const __nv_bfloat16*, __nv_bfloat16*); }\n")
    cu = out / (path.stem + ".cu")
    cu.write_text(src)
    cubin = out / (path.stem + ".cubin")
    subprocess.run([nvcc, "-O3", "-std=c++17", "-arch=sm_90a", "-cubin", "-DGEMV_DEVICE_ONLY", "-o", str(cubin), str(cu)], check=True)
    res = subprocess.run(["cuobjdump", "-res-usage", str(cubin)], check=True, capture_output=True, text=True).stdout
    sass = subprocess.run(["cuobjdump", "-sass", str(cubin)], check=True, capture_output=True, text=True).stdout
    (out / (path.stem + ".sass")).write_text(sass)
    funcs = re.split(r"\n\s*Function : ", sass)
    for f in funcs[1:]:
        name = f.split("\n", 1)[0]
        if "cgemv40_kernel" in name and "ILi128ELi2E" in name:
            ops = collections.Counter(m.group(1) for m in re.finditer(r"/\*[0-9a-f]{4}\*/\s+(?:@!?P\d\s+)?([A-Z0-9_]+)", f))
            print(path.name, name)
            print("  instructions", sum(ops.values()))
            print("  " + " ".join(f"{k} {v}" for k, v in sorted(ops.items())))
    for line in res.splitlines():
        if "cgemv40_kernel" in line and "ILi128ELi2E" in line:
            print("  " + line.strip())
        if line.strip().startswith("REG:") and res.splitlines()[res.splitlines().index(line) - 1].find("cgemv40_kernelILi128ELi2E") >= 0:
            print("  " + line.strip())


if __name__ == "__main__":
    main()
