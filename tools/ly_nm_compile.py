"""Compile the captured non-MoE sources to cubins on the CPU and write SASS, resource usage and a per-kernel summary."""
import collections, concurrent.futures as cf, json, os, re, subprocess, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch.utils.cpp_extension as cx
import tvm_ffi

NVCC = "/usr/local/cuda-13.0/bin/nvcc"
OBJ = "/usr/local/cuda-13.0/bin/cuobjdump"
SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
TORCH_INC = []
for p in cx.include_paths(device_type="cuda"):
    TORCH_INC += ["-isystem", p]
TORCH_INC += ["-isystem", "/usr/include/python3.12"]
TVM_INC = ["-isystem", os.path.join(os.path.dirname(tvm_ffi.__file__), "include")]
TORCH_D = ["-DTORCH_API_INCLUDE_EXTENSION_H", "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
           "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr", "-std=c++20"]
JOBS = {
    "bfp__get_ext": "torch", "_gle_ext": "torch", "_de_ext": "torch", "_cg_build": "torch", "_router_fp32_ext": "torch",
    "_sl_cluster_module": "tvm", "_pdl8_graft_glue_module": "tvm", "_graft_b2_module": "tvm",
}


def gencode(flags):
    return [] if any(f.startswith("-gencode") for f in flags) else ["-gencode=arch=compute_90,code=sm_90"]


def one(key):
    meta = json.load(open(os.path.join(SRC, key + ".json")))
    flags = [f for f in meta.get("extra_cuda_cflags", []) if f != "-lineinfo"]
    src = open(os.path.join(SRC, key + ".cu")).read()
    cu = os.path.join(OUT, key + ".cu")
    if JOBS[key] == "torch":
        pre = "#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n"
        open(cu, "w").write(pre + src)
        cmd = [NVCC, "-cubin", "-o", os.path.join(OUT, key + ".cubin"), f"-DTORCH_EXTENSION_NAME={meta['name']}",
               *TORCH_D, *TORCH_INC, *flags, *gencode(flags), cu]
    else:
        open(cu, "w").write(src)
        cmd = [NVCC, "-cubin", "-o", os.path.join(OUT, key + ".cubin"), "-std=c++17", "-O2", *TVM_INC, *flags,
               *gencode(flags), cu]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        return key, "nvcc failed: " + r.stderr[-3000:]
    cub = os.path.join(OUT, key + ".cubin")
    sass = subprocess.run([OBJ, "-sass", cub], capture_output=True, text=True).stdout
    res = subprocess.run([OBJ, "-res-usage", cub], capture_output=True, text=True).stdout
    open(os.path.join(OUT, key + ".sass"), "w").write(sass)
    open(os.path.join(OUT, key + ".res"), "w").write(res)
    open(os.path.join(OUT, key + ".cmd"), "w").write(" ".join(cmd))
    return key, "ok"


INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);")


def summary(key):
    sass = open(os.path.join(OUT, key + ".sass")).read()
    res = open(os.path.join(OUT, key + ".res")).read()
    rs = {}
    for m in re.finditer(r"Function (\S+):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", res):
        rs[m.group(1)] = f"reg={m.group(2)} stack={m.group(3)} local={m.group(5)} shared={m.group(4)}"
    cur, ks = None, collections.OrderedDict()
    for line in sass.splitlines():
        mf = re.search(r"Function : (\S+)", line)
        if mf:
            cur = mf.group(1)
            ks[cur] = collections.Counter()
            continue
        mi = INS.search(line) if cur else None
        if mi:
            op = mi.group(3)
            pred = "@" if mi.group(2) else ""
            if op.startswith(("STG", "ST.", "STS", "RED", "ATOM", "LDG", "MUFU", "LDS", "STAS", "LD.", "BRA", "UBLKCP", "UTMALDG")):
                ks[cur][pred + op] += 1
            ks[cur]["_n"] += 1
    with open(os.path.join(OUT, key + ".summary"), "w") as f:
        for k, c in ks.items():
            f.write(f"{k}\n  {rs.get(k, '')} ins={c.pop('_n', 0)}\n  " +
                    " ".join(f"{o}:{n}" for o, n in sorted(c.items())) + "\n")


keys = sys.argv[3:] or list(JOBS)
with cf.ThreadPoolExecutor(len(keys)) as ex:
    for key, st in ex.map(one, keys):
        print(key, st[:3000], flush=True)
        if st == "ok":
            summary(key)
