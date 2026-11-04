#!/usr/bin/env python3
"""ly_bfq: ly_stk3 with the qkv GEMM at T = 128 (verify 16) on muse_bf16_proj's wide kernel wd b1n128c3s8h
(TMA + wgmma, K split over a 3-CTA cluster, 8-stage ring) instead of the pinned cuBLASLt plan, the kernel
triggering griddepcontrol.launch_dependents at entry, and the attention launched behind it as a PDL secondary.

Measured before the port (docs/lanes/bf16.md): standalone at m = 128 the wd pick is 18.58 us against 19.82 us
for the Lt plan; the box-only prototype (tools/dl_ly_bfq_proto.py) gave verify 16 sliding -1.36 us (18/24).

The CUDA source is muse_decode_layer/submission.py's inlined copy (_SKP_SRC + _SKP_GEMV_SRC + _SKP_WD_SRC, built
there by tools/dl_proj_inline.py), renamed _BFQW_, plus one line at the top of wd_kernel's body. qkv is the first
kernel of the layer and is launched plainly, so it needs no wait; the attention waits (pdl_wait) before its first
qkv read, and griddepcontrol.wait returns only after the whole primary grid has completed and flushed (PTX ISA
9.7.13.13). A dependent grid launches only after every primary CTA has triggered or exited, so attention CTAs never
take an SM from a wd CTA that has not started.

prepare self-tests the kernel at 128 tokens against FP64 with muse_decode_layer/submission.py's gate SKP_TOL
(relative L2 within 5e-3, line `SKP_TOL = 5e-3`) and against the Lt plan's output, then runs the PDL8H stress test
(PDL8_STRESS replays, _pdl8h_guard) at 16 x 8 sliding and full; any failure turns the swap off.

    python3 tools/dl_ly_bfq_patch.py   # writes muse_decode_layer/cand/ly_bfq.py
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk3.py"
DST = ROOT / "muse_decode_layer/cand/ly_bfq.py"
PIN = "a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6"
WD_HOST = ROOT / "muse_decode_layer/submission.py"
WD_NAMES = ("_SKP_SRC", "_SKP_GEMV_SRC", "_SKP_WD_SRC", "_SKP_WD_GEOM", "_SKP_CPP_SRC")

TRIGGER_OLD = "    bool dead = false;\n\n    if (tid == WG * 128) {"
TRIGGER_NEW = ("    bool dead = false;\n"
               "    asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n\n"
               "    if (tid == WG * 128) {")

EDITS = [
    ('pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"]) or _pdl8_edge("attn")} if ATT_HAS_PDL else {}',
     'pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"]) or _pdl8_edge("attn") or _bfq_attn_edge()} if ATT_HAS_PDL else {}',
     1),
]

TAIL = r'''


# ---- bfq (tools/dl_ly_bfq_patch.py): qkv at T = 128 on wd b1n128c3s8h, attention behind it as a PDL secondary ----
BFQ_T = 128
BFQ_GEOM = "b1n128c3s8h"
BFQ_TOL = 5e-3
_BFQ = {"on": False, "attn": True, "plan": None, "run": None}


def _bfq_attn_edge():
    return _BFQ["on"] and _BFQ["attn"]


def _bfqw_load():
    src = _BFQW_SRC + _BFQW_GEMV_SRC + _BFQW_WD_SRC
    kwargs = dict(name="okc_ly_bfq_wd", cpp_sources=_BFQW_CPP_SRC, cuda_sources=src, functions=["wd_fwd"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
    try:
        return load_inline(no_implicit_headers=True, **kwargs)
    except TypeError:
        kwargs["cuda_sources"] = src.replace("#include <ATen/core/Tensor.h>", "#include <torch/types.h>")
        return load_inline(**kwargs)


def _bfq_runner(ext, w, diag):
    cid = list(_BFQW_WD_GEOM).index(BFQ_GEOM)
    n, k = w.shape

    def run(x):
        out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
        ext.wd_fwd(cid, w, x, out, n, k, diag)
        return out
    return run


_bfq_prior_lt_run = _graft_lt_run
def _graft_lt_run(x, plan):
    if _BFQ["on"] and plan is _BFQ["plan"]:
        return _BFQ["run"](x)
    return _bfq_prior_lt_run(x, plan)


_bfq_prior_kernel = kernel_fn
def kernel_fn(data, state):
    _BFQ["on"] = data["config"]["tokens"] == BFQ_T and bool(state.get("bfq_on", False))
    _BFQ["attn"] = bool(state.get("bfq_attn", True))
    try:
        return _bfq_prior_kernel(data, state)
    finally:
        _BFQ["on"] = False


def _bfq_selftest(st, w):
    plan = st.get("graft_lt", {}).get(("qkv", BFQ_T))
    n, k = w.shape
    if plan is None or plan["n"] != n or plan["fp32"]:
        raise RuntimeError(f"qkv {BFQ_T} Lt plan missing or unexpected")
    diag = torch.zeros(16, dtype=torch.int32, device=w.device)
    run = _bfq_runner(_bfqw_load(), w, diag)
    gen = torch.Generator(device=w.device).manual_seed(1357)
    x = torch.randn((BFQ_T, k), generator=gen, device=w.device).to(torch.bfloat16)
    out = run(x)
    lt = _bfq_prior_lt_run(x, plan)
    torch.cuda.synchronize()
    ref = (x.double() @ w.double().T).to(torch.bfloat16).double()
    err = float((out.double() - ref).norm() / ref.norm())
    err_lt = float((lt.double() - ref).norm() / ref.norm())
    vs_lt = float((out.double() - lt.double()).norm() / lt.double().norm())
    print(f"BFQ selftest {BFQ_GEOM} m={BFQ_T}: rel L2 {err:.2e} (Lt {err_lt:.2e}, wd vs Lt {vs_lt:.2e}), "
          f"diag {int(diag[0])}", flush=True)
    if not (err <= BFQ_TOL and vs_lt <= BFQ_TOL and int(diag[0]) == 0):
        raise RuntimeError("selftest failed")
    return plan, run


def _bfq_guard(st):
    dev = st["qkv"].device
    gen = torch.Generator(device=dev).manual_seed(27182)
    for B, layer in ((16, "sliding"), (16, "full")):
        T = 8 * B
        data = _pdl8_data(B, 8, layer, gen, dev)
        off = dict(st, bfq_on=False)
        on = dict(st, bfq_on=True)
        ids = _pdl8_flat(kernel_fn(data, off))[-1]
        if ids.dtype in (torch.int32, torch.int64) and ids.numel() == 8 * T:
            counts = torch.bincount(ids.flatten().long().clamp_min(0) // 32, minlength=8)
            data["local_start"].fill_(32 * int(counts.argmax()))
        kernel_fn(data, on)
        torch.cuda.synchronize()
        g_off, o_off = _pdl8_graph(lambda: kernel_fn(data, off))
        g_on, o_on = _pdl8_graph(lambda: kernel_fn(data, on))
        rows = _pdl8_time([g_off, g_on])
        d = sorted(r[0] - r[1] for r in rows)
        print(f"BFQ_AB T={T} {layer}: off {sorted(r[0] for r in rows)[len(rows) // 2]:.2f} on "
              f"{sorted(r[1] for r in rows)[len(rows) // 2]:.2f} us, median paired saving {d[len(d) // 2]:.2f}, "
              f"wins {sum(1 for v in d if v > 0)}/{len(d)}", flush=True)
        # Each graph is checked against its own first output: the two arms round differently, and the guard
        # is about replay-to-replay drift under the early trigger.
        g_off.replay()
        g_on.replay()
        torch.cuda.synchronize()
        ref_off = [t.clone() for t in o_off]
        ref_on = [t.clone() for t in o_on]
        off_max, off_bad = _pdl8_drift(g_off, o_off, ref_off, PDL8_STRESS)
        on_max, on_bad = _pdl8_drift(g_on, o_on, ref_on, PDL8_STRESS)
        ok = _pdl8h_guard(st, T, layer, off_max, off_bad, on_max, on_bad)
        print(f"BFQ PDL8H T={T} {layer}: off max {off_max} bad {off_bad}, on max {on_max} bad {on_bad} -> "
              f"{'pass' if ok else 'FAIL'}", flush=True)
        del g_off, g_on, o_off, o_on, data
        torch.cuda.empty_cache()
        if not ok:
            return False
    return True


_bfq_prior_prepare = prepare
def prepare(weights):
    st = _bfq_prior_prepare(weights)
    st["bfq_on"] = False
    try:
        _BFQ["plan"], _BFQ["run"] = _bfq_selftest(st, weights["qkv_gate"].contiguous())
        st["bfq_on"] = _bfq_guard(st)
    except Exception as e:
        print("BFQ off:", repr(e)[:400], flush=True)
        st["bfq_on"] = False
        torch.cuda.synchronize()
    print(f"BFQ enabled {st['bfq_on']}", flush=True)
    return st
'''


def wd_block():
    text = WD_HOST.read_text()
    tree = ast.parse(text)
    seg = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in WD_NAMES:
                seg[name] = ast.get_source_segment(text, node)
    missing = [n for n in WD_NAMES if n not in seg]
    if missing:
        sys.exit(f"missing in {WD_HOST}: {missing}")
    wd = seg["_SKP_WD_SRC"]
    if wd.count(TRIGGER_OLD) != 1 or "griddepcontrol" in wd:
        sys.exit("wd_kernel trigger anchor not unique or griddepcontrol already present")
    seg["_SKP_WD_SRC"] = wd.replace(TRIGGER_OLD, TRIGGER_NEW)
    return "\n\n".join(seg[n] for n in WD_NAMES).replace("_SKP_", "_BFQW_")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    if "_BFQW_" in text or "_bfq" in text:
        sys.exit("base already carries bfq names")
    for old, new, n in EDITS:
        if text.count(old) != n:
            sys.exit(f"edit anchor found {text.count(old)} times, expected {n}: {old[:80]!r}")
        text = text.replace(old, new)
    text = ("# ly_bfq: ly_stk3 with qkv at T = 128 on wd b1n128c3s8h and the attention behind it as a PDL secondary "
            "(tools/dl_ly_bfq_patch.py).\n" + text + "\n\n" + wd_block() + TAIL)
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
