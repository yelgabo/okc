"""Import the layer file with CUDA hidden and recording load_inline fakes; dump each non-MoE build's CUDA source and flags."""
import importlib.util, json, os, sys, types
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.utils.cpp_extension as cx
torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
GOT = []


class Ext:
    def __getattr__(self, n):
        return lambda *a, **k: 0


def rec(*a, **kw):
    names = ["name", "cpp_sources", "cuda_sources"]
    for i, v in enumerate(a):
        kw[names[i]] = v
    GOT.append(kw)
    return Ext()


cx.load_inline = rec
try:
    import tvm_ffi.cpp
    tvm_ffi.cpp.load_inline = rec
except Exception as e:
    mm = types.ModuleType("tvm_ffi.cpp")
    mm.load_inline = rec
    sys.modules.setdefault("tvm_ffi", types.ModuleType("tvm_ffi"))
    sys.modules["tvm_ffi.cpp"] = mm
    print("tvm_ffi fake", e)

path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
spec = importlib.util.spec_from_file_location("cand", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.load_inline = rec
builders = ["_att7", "bfp__get_ext", "_gle_ext", "_de_ext", "_cg_build", "_router_fp32_ext",
            "_graft_glue_module", "_graft_b2_module", "_sl_cluster_module", "_pdl8_cluster_module",
            "_pdl8_graft_glue_module"]
for b in builders:
    for g in ("_ATT7", "_CG_EXT", "_ROUTER_FP32_EXT", "_GLE_EXT", "_DE_EXT", "bfp__ext", "_PDL8_GLUE", "_PDL8_GRAFT_GLUE"):
        if hasattr(m, g):
            setattr(m, g, None)
    n0 = len(GOT)
    try:
        getattr(m, b)()
    except Exception as e:
        print(b, "raised", repr(e)[:300])
    for i, kw in enumerate(GOT[n0:]):
        key = f"{b}{'' if i == 0 else i}"
        src = kw.get("cuda_sources")
        if isinstance(src, (list, tuple)):
            src = "\n".join(src)
        open(os.path.join(out, key + ".cu"), "w").write(src or "")
        cpp = kw.get("cpp_sources")
        if isinstance(cpp, (list, tuple)):
            cpp = "\n".join(cpp)
        open(os.path.join(out, key + ".cpp"), "w").write(cpp or "")
        meta = {k: v for k, v in kw.items() if k not in ("cuda_sources", "cpp_sources")}
        json.dump(meta, open(os.path.join(out, key + ".json"), "w"), indent=1, default=str)
        print(key, kw.get("name"), kw.get("extra_cuda_cflags"), len(src or ""))
