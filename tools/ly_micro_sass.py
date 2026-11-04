#!/usr/bin/env python3
"""SASS of a layer file's five fused-MoE builds, compiled on the box CPU with the file's own load_inline arguments.

The file is imported with CUDA hidden; load_inline is replaced by a recorder, so _ma_ext(defines) yields the exact
cuda source and extra_cuda_cflags the judge build would get (production suffix: PRMTSC, SC6W on WGS builds, DQ7 from
_ma_ext). Each build is compiled to a cubin with the torch include paths of the extension build and reported:
registers, local memory (spill) bytes, and per kernel the instruction count of every loop that holds an HMMA / HGMMA
(the unit loops), with its opcode mix.

    CUDA_VISIBLE_DEVICES= python tools/ly_micro_sass.py muse_decode_layer/cand/<file>.py OUTDIR [build ...]
"""
import collections
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402
import torch.utils.cpp_extension as cx  # noqa: E402

torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
NVCC = "/usr/local/cuda-13.0/bin/nvcc"
CUOBJDUMP = "/usr/local/cuda-13.0/bin/cuobjdump"


def builds(m):
    sfx = tuple(m._MA_SFX_IF_PRMT)
    wgs = ("-DOKC_SC6W=1",)
    return {
        "tiny": tuple(m.MOE_TINY_DEFINES) + sfx,
        "tiny2": tuple(m.MOE_TINY2_DEFINES) + sfx,
        "mid": tuple(m.MOE_MID_DEFINES) + sfx,
        "b4": tuple(m.MOE_DEFINES) + ("-DOKC_B4FIN=1",) + wgs + sfx,
        "main": tuple(m._main_defines(sfx)) + tuple(getattr(m, "MAIN_EXTRA_DEFINES", ())) + wgs + sfx + tuple(os.environ.get("LY_SASS_MAIN_EXTRA", "").split()),
    }


def capture(path):
    spec = importlib.util.spec_from_file_location("cand", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    got = {}

    class Ext:
        def okc_set_pdl(self, *a):
            pass

    def rec(**kw):
        got.update(kw)
        return Ext()

    m.load_inline = rec
    out = {}
    for k, d in builds(m).items():
        got.clear()
        m._MA_EXTS.clear()
        m._ma_ext(d)
        out[k] = (got["name"], got["cuda_sources"], list(got["extra_cuda_cflags"]))
    return out


def compile_one(outdir, key, name, src, flags):
    cu = os.path.join(outdir, f"{key}.cu")
    with open(cu, "w") as f:
        f.write("#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n" + src)
    inc = []
    for p in cx.include_paths(device_type="cuda"):
        inc += ["-isystem", p]
    inc += ["-isystem", "/usr/include/python3.12"]
    flags = [f for f in flags if f != "-lineinfo"]
    cubin = os.path.join(outdir, f"{key}.cubin")
    cmd = [NVCC, "-cubin", "-o", cubin, f"-DTORCH_EXTENSION_NAME={name}",
           "-DTORCH_API_INCLUDE_EXTENSION_H", "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
           "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr",
           "-std=c++20", *inc, *flags, cu]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{key}: nvcc failed\n{r.stderr[-4000:]}")
    sass = subprocess.run([CUOBJDUMP, "-sass", cubin], capture_output=True, text=True, check=True).stdout
    res = subprocess.run([CUOBJDUMP, "-res-usage", cubin], capture_output=True, text=True, check=True).stdout
    open(os.path.join(outdir, f"{key}.sass"), "w").write(sass)
    return key, sass, res


INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);")


def analyze(sass):
    kernels = {}
    cur = None
    for line in sass.splitlines():
        mf = re.search(r"Function : (\S+)", line)
        if mf:
            cur = mf.group(1)
            kernels[cur] = []
            continue
        mi = INS.search(line) if cur else None
        if mi and mi.group(3) != "NOP":
            kernels[cur].append((int(mi.group(1), 16), mi.group(3), mi.group(4)))
    out = {}
    for k, ins in kernels.items():
        at = {a: i for i, (a, _, _) in enumerate(ins)}
        loops = []
        for i, (a, op, args) in enumerate(ins):
            if op.startswith("BRA"):
                t = re.search(r"0x([0-9a-f]+)", args)
                if t and int(t.group(1), 16) <= a and int(t.group(1), 16) in at:
                    s0 = at[int(t.group(1), 16)]
                    body = [o for _, o, _ in ins[s0:i + 1]]
                    mma = sum(o.startswith(("HMMA", "HGMMA")) for o in body)
                    if mma:
                        mix = collections.Counter(o.split(".")[0] for o in body)
                        loops.append({"start": hex(ins[s0][0]), "end": hex(a), "n": len(body), "mma": mma,
                                      "mix": dict(mix.most_common())})
        tot = collections.Counter(o.split(".")[0] for _, o, _ in ins)
        out[k] = {"n": len(ins), "loops": loops, "stg": sum(o.startswith("STG") for _, o, _ in ins),
                  "mufu": sum(o.startswith("MUFU") for _, o, _ in ins), "mix": dict(tot.most_common(30))}
    return out


def resources(res):
    out = {}
    for m in re.finditer(r"Function (\S+):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", res):
        out[m.group(1)] = {"reg": int(m.group(2)), "stack": int(m.group(3)), "local": int(m.group(5))}
    return out


def main():
    path, outdir = sys.argv[1], sys.argv[2]
    want = sys.argv[3:] or ["tiny", "tiny2", "mid", "b4", "main"]
    os.makedirs(outdir, exist_ok=True)
    cap = capture(path)
    report = {}
    with cf.ThreadPoolExecutor(len(want)) as ex:
        futs = [ex.submit(compile_one, outdir, k, *cap[k]) for k in want]
        for f in futs:
            k, sass, res = f.result()
            r, a = resources(res), analyze(sass)
            report[k] = {"src_md5": hashlib.md5(cap[k][1].encode()).hexdigest(), "name": cap[k][0],
                         "flags": cap[k][2], "kernels": {n: {**a.get(n, {}), **r.get(n, {})} for n in a}}
    json.dump(report, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    for k in want:
        for n, v in report[k]["kernels"].items():
            if "moe_fused" not in n:
                continue
            ls = "; ".join(f"loop@{l['start']} n={l['n']} mma={l['mma']}" for l in v["loops"])
            print(f"{k:5s} {n[:40]:40s} reg={v.get('reg')} local={v.get('local')} stack={v.get('stack')} "
                  f"ins={v['n']} stg={v['stg']} mufu={v['mufu']} | {ls}")


if __name__ == "__main__":
    main()
