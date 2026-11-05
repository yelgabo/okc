#!/usr/bin/env python3
"""Port the decode layer's projection sk2 path (keell0renz's BF16 sk2 kernel as embedded in the layer file, plus the
layer's narrow qkv build for m <= 8) into the standalone muse_bf16_proj file.

Text patch: every anchor must match exactly once, or the tool stops.  The base file is copied unchanged and the port
is appended after it, overriding prepare and kernel_fn only for the (layer, m) keys named in --use.

  python3 tools/dl_bfp_port_patch.py --base .build/repeats/bf16_base.py \
      --layer muse_decode_layer/cand/ly_stk7c.py --use all --out muse_bf16_proj/cand/bf16_port.py

--use: comma list of layer:m:mode with mode s (layer's sk2 geometry) or n (narrow qkv, m <= 8 only); "all" puts every
layer at m 1, 8, 16, 32 on sk2 with qkv m <= 8 narrow; "alls" is the same with qkv on the wide sk2 geometry.
"""
import argparse
import hashlib
import sys

LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")

# (start line, end line exclusive) anchors in the layer file
BLOCKS = [
    ("bfp_CUDA_SRC = r'''\n", "bfp_WG_HELPERS = r\"\"\"\n"),
    ("bfp_MAP_SRC = r'''\n", "bfp_GEMM_SRC = r'''\n"),
    ("bfp_SK2_SRC = r'''\n", "bfp_WIDE_SRC = r'''\n"),
    ("bfp_CPP_SRC = \"\"\"\n", "bfp_w_rows = {"),
    ("bfp_SK2_CONFIGS = [", "bfp_WIDE_BF16 = False"),
    ("def bfp__sk2_name(layer, m):\n", "def bfp__run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):\n"),
    ("def _de_source():\n", "def _de_ext():\n"),
]

BASE_ANCHORS = ["def prepare(weights):\n", "def kernel_fn(data, state):\n", "from torch.utils.cpp_extension import load_inline\n"]

GLUE = r'''

def _port_sk2_source():
    lines = []
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        for tokr in (8, 16, 32):
            if bfp_SK2_PICK.get((layer, tokr)) != name:
                continue
            hl, nstage = bfp__sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            ot = "float" if hl else "__nv_bfloat16"
            nacc = 1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC
            lines.append(f"if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, "
                         f"{nstage}, {str(not hl).lower()}, {str(hl).lower()}, {nacc}, {CL}, {ot}>(Ws, x, out, ptab, "
                         f"pmax, base4, M, N, K, dg, dbg); return; }}")
    source = bfp_SK2_SRC.replace("@@SK2_DISPATCH@@", "\n  ".join(lines)).replace("@@SK2F_DISPATCH@@", "")
    return bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", "") + bfp_MAP_SRC + source


def _port_load(source, prefix, arch):
    import hashlib
    tag = hashlib.md5(source.encode()).hexdigest()[:12]
    source = source.replace("sk2_kernel", "sk2_kernel_" + prefix + tag).replace("launch_sk2", "launch_sk2_" + prefix + tag)
    cpp = bfp_CPP_SRC[bfp_CPP_SRC.index("void sk2("):bfp_CPP_SRC.index("void sk2f(")] + "void sk2_set_pdl(int64_t n);"
    return load_inline(f"okc_bfport_{prefix}_{tag}", "#include <torch/extension.h>\n" + cpp, cuda_sources=source,
                       functions=["sk2", "sk2_set_pdl"], no_implicit_headers=True,
                       extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"], verbose=False)


def _port_runner(ext, ent, layer, m, n, k, dummy):
    ci, stream, tab, pmax, b4, n_rows = ent
    odt = torch.float32 if layer == "router" else torch.bfloat16
    tokr = bfp__sk2_tokr(m)
    sk2 = ext.sk2

    def run(x):
        out = torch.empty((m, n), dtype=odt, device=x.device)
        sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, tokr, 0, dummy)
        return out
    return run


def _port_prepare(weights, state):
    t0 = time.time()
    if not _PORT_USE or torch.cuda.get_device_capability()[0] < 9:
        return
    dev = weights["o"].device
    st = {"sm90": True, "dummy": torch.zeros(2, dtype=torch.int32, device=dev)}
    bfp_sk2_prepare({k: v.contiguous() for k, v in weights.items()}, st)
    exts = {}
    if any(v == "s" for v in _PORT_USE.values()):
        exts["s"] = _port_load(_port_sk2_source(), "w", "90a")
    if any(v == "n" for v in _PORT_USE.values()):
        exts["n"] = _port_load(_de_source(), "n", "90")
    for e in exts.values():
        e.sk2_set_pdl(0)
    kdim = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
    full = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
    g = torch.Generator(device=dev)
    g.manual_seed(1234)
    for (layer, m), mode in sorted(_PORT_USE.items()):
        n = 1024 if (layer == "unpooler" and m <= 8) else full[layer]
        name = bfp__sk2_name(layer, m)
        ent = st["sk2"].get(name)
        if ent is None or ent[5] < n:
            print(f"bf16-port: {layer} m={m} no sk2 entry, base kept")
            continue
        run = _port_runner(exts[mode], ent, layer, m, n, kdim[layer], st["dummy"])
        x = torch.randn((m, kdim[layer]), generator=g, device=dev).to(torch.bfloat16)
        ref = x.float() @ weights[layer][:n].float().t()
        got = run(x).float()
        torch.cuda.synchronize()
        err = float((got - ref).abs().max()) / max(float(ref.abs().max()), 1e-6)
        if not err < 2e-2:
            print(f"bf16-port: {layer} m={m} mode={mode} rel err {err:.3g}, base kept")
            continue
        state["port_pick"][(layer, m)] = run
    state["port_sk2"] = st
    print(f"bf16-port: {len(state['port_pick'])} sk2 picks in {time.time() - t0:.1f} s")


_port_base_prepare = prepare
_port_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _port_base_prepare(weights)
    state["port_pick"] = {}
    try:
        _port_prepare(weights, state)
    except Exception as exc:
        print(f"bf16-port: sk2 off ({type(exc).__name__}: {str(exc)[-800:]})")
        state["port_pick"] = {}
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    run = state["port_pick"].get((cfg["layer"], cfg["m"]))
    if run is not None:
        return run(data["x"].contiguous())
    return _port_base_kernel_fn(data, state)
'''


def parse_use(spec):
    if spec in ("all", "alls"):
        use = {(l, m): "s" for l in LAYERS for m in (1, 8, 16, 32)}
        if spec == "all":
            use[("qkv_gate", 1)] = use[("qkv_gate", 8)] = "n"
        return use
    use = {}
    for item in filter(None, spec.split(",")):
        layer, m, mode = item.split(":")
        m = int(m)
        assert layer in LAYERS and 1 <= m <= 32 and mode in ("s", "n"), item
        assert mode == "s" or (layer == "qkv_gate" and m <= 8), item
        use[(layer, m)] = mode
    return use


def once(text, anchor, what):
    c = text.count(anchor)
    if c != 1:
        sys.exit(f"anchor {anchor[:50]!r} in {what}: {c} matches, need 1")
    return text.index(anchor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--use", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = open(a.base).read()
    layer = open(a.layer).read()
    for anc in BASE_ANCHORS:
        once(base, anc, "base")
    if not base.startswith("# okc: problem=muse_bf16_proj"):
        sys.exit("base header is not muse_bf16_proj")
    parts = []
    for s, e in BLOCKS:
        i = once(layer, s, "layer")
        j = once(layer, e, "layer")
        assert j > i, s
        parts.append(layer[i:j].rstrip() + "\n")
    use = parse_use(a.use)
    head = ("\n\n# bf16-port: the decode layer's projection sk2 path at m <= 32.  Kernel, packing and geometries come from\n"
            "# keell0renz's BF16 submission as embedded in muse_decode_layer/cand/ly_stk7c.py; the narrow qkv build\n"
            "# (_de_source, m <= 8) is the layer's ly_q1n change.  Weights stay BF16 (12-bit lossless packing with exact\n"
            "# escape corrections); the router keeps FP32 weights as a BF16 hi + lo pair.\n"
            "import os\n\n")
    usetxt = "_PORT_USE = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in sorted(use.items())) + "}\n"
    out = base.rstrip() + "\n" + head + "\n\n".join(parts) + "\n\n" + usetxt + GLUE
    if len(out.encode()) >= 1_000_000:
        sys.exit(f"output {len(out.encode())} bytes, limit 1,000,000")
    compile(out, a.out, "exec")
    open(a.out, "w").write(out)
    print(a.out, len(out.encode()), "bytes sha256", hashlib.sha256(out.encode()).hexdigest(), "keys", len(use))


if __name__ == "__main__":
    main()
