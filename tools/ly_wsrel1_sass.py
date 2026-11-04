#!/usr/bin/env python3
"""SASS of the moe_helper0 doneA credit in ly_mpdl against ly_wsrel1 (acq_rel) and ly_wsrel1_sc, device build.

python tools/ly_wsrel1_sass.py [--out DIR]     (on the box, venv active)
Extracts _MA_ALT_SRCS['moe_helper0'] from each file, compiles it with nvcc for sm_90a under the TINY and MID define
sets plus the flags _ma_ext adds (the per-weights suffix is left out), counts MEMBAR / CCTL opcodes and prints the
opcode-level diff against ly_mpdl, with a few lines of context, so the credit's fence placement can be read.
"""
import collections
import difflib
import pathlib
import re
import subprocess
import sys
import sysconfig

from torch.utils.cpp_extension import include_paths

FILES = {"base": "ly_mpdl", "ar1": "ly_wsrel1", "sc1": "ly_wsrel1_sc"}
COMMON = ["-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_FNST=4", "-DOKC_NSF=3"]
BUILDS = {
    "tiny": ["-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0",
             "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1", "-DOKC_NOSCL=1"],
    "mid": ["-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_CPF=0", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4",
            "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1"],
}
SPLIT = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"


def sass(stem, build, out):
    text = pathlib.Path(f"muse_decode_layer/cand/{stem}.py").read_text()
    i = text.index(SPLIT) + len(SPLIT)
    src = "#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n" + text[i:text.index('\n"""\n', i) + 1]
    cu = out / f"{stem}_{build}.cu"
    cu.write_text(src)
    obj = out / f"{stem}_{build}.o"
    inc = sum((["-I", p] for p in include_paths(device_type="cuda") + [sysconfig.get_paths()["include"]]), [])
    subprocess.run(["nvcc", "-c", "-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", *inc, *COMMON,
                    *BUILDS[build], str(cu), "-o", str(obj)], check=True)
    dump = subprocess.run(["cuobjdump", "-sass", str(obj)], check=True, capture_output=True, text=True).stdout
    (out / f"{stem}_{build}.sass").write_text(dump)
    ops = []
    for line in dump.splitlines():
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;", line)
        if m:
            t = m.group(1).split()
            ops.append("@P " + t[1] if t[0].startswith("@") else t[0])  # registers differ per build: opcodes only
    return ops


def main():
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else pathlib.Path(".build/wsrel1/sass")
    out.mkdir(parents=True, exist_ok=True)
    for build in BUILDS:
        ops = {k: sass(stem, build, out) for k, stem in FILES.items()}
        for k, o in ops.items():
            c = collections.Counter(x.split()[-1] for x in o)
            print(f"SASS {build} {k}: n={len(o)}", {op: n for op, n in sorted(c.items()) if op.startswith(("MEMBAR", "CCTL", "FENCE", "ERRBAR", "WARPSYNC", "ATOM", "RED"))}, flush=True)
        for k in ("ar1", "sc1"):
            print(f"DIFF {build} base -> {k}")
            for line in difflib.unified_diff(ops["base"], ops[k], lineterm="", n=4):
                if not line.startswith(("---", "+++")):
                    print("  " + line)


if __name__ == "__main__":
    main()
