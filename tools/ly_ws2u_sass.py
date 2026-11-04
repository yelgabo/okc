#!/usr/bin/env python3
"""SASS of a layer file's MID fused-MoE build, compiled with nvcc only (no GPU, no CUDA context): the same source,
defines and flags as the file's _ma_ext / _moe_build use for MID (with the PRMTSC suffix prepare picks on the
judge's weights, and DQ7 as the file sets it).

    CUDA_VISIBLE_DEVICES= python tools/ly_ws2u_sass.py muse_decode_layer/cand/<file>.py OUTDIR
writes OUTDIR/<stem>.sass (cuobjdump -sass) and OUTDIR/<stem>.res (cuobjdump -res-usage) and prints ptxas -v.
"""
import importlib.util
import os
import subprocess
import sys
import sysconfig

import torch.utils.cpp_extension as ce

path, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
stem = os.path.splitext(os.path.basename(path))[0]
sys.path.insert(0, os.path.abspath("muse_decode_layer"))
spec = importlib.util.spec_from_file_location("okc_sub_" + stem, path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
defines = tuple(mod.MOE_MID_DEFINES) + tuple(mod._MA_SFX_IF_PRMT)
if mod.DQ7 and "-DOKC_DQ7=1" not in defines:
    defines += ("-DOKC_DQ7=1",)
src = mod.MOE_CUDA_SRC
for d in defines:
    if d.startswith("@"):
        src = mod._MA_ALT_SRCS[d[1:]]
defs = [d for d in defines if not d.startswith("@")]
cu = os.path.join(outdir, stem + ".cu")
with open(cu, "w") as f:
    f.write(src)
inc = [f"-I{p}" for p in ce.include_paths(device_type="cuda")] + ["-I" + sysconfig.get_paths()["include"]]
flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-DOKC_NS=3", "-DOKC_FNST=4", "-DOKC_NSF=3", *defs,
         "-std=c++20", "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
         "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr",
         "-DTORCH_EXTENSION_NAME=sasschk", "-DTORCH_API_INCLUDE_EXTENSION_H", "-D_GLIBCXX_USE_CXX11_ABI=1",
         "-Xptxas", "-v"]
cub = os.path.join(outdir, stem + ".cubin")
r = subprocess.run(["nvcc", "-cubin", *flags, *inc, cu, "-o", cub], capture_output=True, text=True)
print(stem, "defines", defs)
print("\n".join(line for line in r.stderr.splitlines() if "registers" in line or "error" in line or "moe_fused" in line))
if r.returncode:
    sys.exit(r.stderr[-4000:])
for opt, ext in (("-sass", "sass"), ("-res-usage", "res")):
    with open(os.path.join(outdir, f"{stem}.{ext}"), "w") as f:
        f.write(subprocess.run(["cuobjdump", opt, cub], capture_output=True, text=True).stdout)
