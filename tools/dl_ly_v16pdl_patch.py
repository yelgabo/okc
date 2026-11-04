#!/usr/bin/env python3
"""ly_v16pdl: at T = 128 / 256 (verify 16 / 32) the three Triton / Gluon kernels behind a cuBLAS GEMM and the
attention behind the qkv GEMM launch as PDL secondaries.

Base: muse_decode_layer/cand/ly_stk1.py (SHA256 d7db12a1..., checked). At T = 128 / 256 `_pdl(T)` is False
(PDL = False) and T is not in PDL8_T, so every edge of the verify 16 / 32 chain is a plain launch; the box timeline
of ly_stk1 shows +0.26 to +0.93 us start gaps on all eight (docs/runs-decode-layer.md, "Verify 16 / 32 edges and
unpooler stages (box)"). Four of the nine kernels are cuBLASLt nvjet GEMMs, which take no launch attribute and
execute no griddepcontrol, so they stay plain launches and cannot be secondaries; the MoE stays a plain cooperative
launch. The four edges whose secondary is our own kernel and waits at entry are opened:

  qkv (nvjet) -> attention (attn_kernel<8, false, 3>): `_pdl8_decode_attention7` passes pdl=True; the kernel runs
      `pdl_wait()` before its first qkv read (the same binary PDL8D opens at T = 8);
  o (nvjet) -> b1 `_graft_row_boundary`, unpooler (nvjet) -> b2 `_graft_row_boundary`: `_gdc_wait()` is the
      kernel's first statement;
  pooler + router (nvjet) -> `_graft_route`: `_gdc_wait()` before any load (EARLY = False).

An nvjet primary has no explicit trigger, so a secondary launches only when every primary CTA has exited, and its
griddepcontrol.wait returns only after the primary grid has completed and its memory is visible (PTX ISA 8.7,
9.7.13.13). Every kernel before the primary was a plain launch, so it completed before the primary started. No
secondary writes global memory before its wait. What the edge can save is the launch latency and the completion
flush between the primary's last CTA exit and the secondary's first instruction.

With --pre, the row boundaries at T > 32 (b1 and b2) also take the `_graft_row_boundary_pre` body (the T = 32
form): the norm weights, alpha and beta (prepare-time constants) load before the wait; the branch and stream rows
after it.

Switch: `V16PDL_T = (128, 256)`; `kernel_fn` sets `_V16PDL` from `state["v16pdl_on"][T]`. The in-prepare guard
(`_v16_setup`, after the PDL8 setup) runs at 16 x 8 sliding and 32 x 8 full the PDL8H test: 2000 replays per arm
of the PDL-off and PDL-on graphs against the PDL-off output, ids mismatches within 2 x off + 16 and every output's
max relative difference within max(2e-2, off); a failure turns that T off. A paired time A/B is printed at all four
cells (V16_AB) and gates nothing.

    python3 tools/dl_ly_v16pdl_patch.py [--pre] [--base F] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
PRE = "--pre" in sys.argv
EDGES = sys.argv[sys.argv.index("--edges") + 1].split(",") if "--edges" in sys.argv else None
DST = ROOT / ("muse_decode_layer/cand/ly_v16pre.py" if PRE else "muse_decode_layer/cand/ly_v16pdl.py")
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

EDITS = [
    ('        _graft_row_boundary[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],\n'
     '                                  g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 16 if T > 192 else 32,\n'
     '                                  num_warps=16 if T > 192 else 32, launch_pdl=False)',
     '        (_graft_row_boundary_pre if (_v16e("b2" if post else "b1") and V16PRE) else _graft_row_boundary)[(T,)](\n'
     '                                  br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],\n'
     '                                  g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 16 if T > 192 else 32,\n'
     '                                  num_warps=16 if T > 192 else 32, launch_pdl=_v16e("b2" if post else "b1"))', 1),
    ('PACKED=1 if T > 32 else 0, num_warps=4 if T > 32 else 1, launch_pdl=False)',
     'PACKED=1 if T > 32 else 0, num_warps=4 if T > 32 else 1, launch_pdl=_v16e("route") and T > 32)', 4),
    ('pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"]) or _pdl8_edge("attn")} if ATT_HAS_PDL else {}',
     'pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"]) or _pdl8_edge("attn") or _v16e("attn")} if ATT_HAS_PDL else {}', 1),
]

TAIL = r'''


# ---- v16pdl (tools/dl_ly_v16pdl_patch.py): T = 128 / 256 edges into our kernels as PDL secondaries ----
V16PDL_T = (128, 256)
V16PRE = __V16PRE__
V16PDL_EDGES = ("attn", "b1", "route", "b2")
V16PDL_DEFAULT = __V16DEF__
_V16PDL = False
_V16E = None


def _v16e(name):
    return _V16PDL and (_V16E is None or name in _V16E)


_v16_prior_kernel = kernel_fn
def kernel_fn(data, state):
    global _V16PDL, _V16E
    T = data["config"]["tokens"]
    _V16PDL = T in V16PDL_T and bool(state.get("v16pdl_on", {}).get(T, False))
    _V16E = state.get("v16pdl_edges", {}).get(T) if _V16PDL else None
    try:
        return _v16_prior_kernel(data, state)
    finally:
        _V16PDL = False
        _V16E = None


def _v16_setup(st):
    dev = st["qkv"].device
    gen = torch.Generator(device=dev).manual_seed(27182)
    st["v16pdl_on"] = {T: True for T in V16PDL_T}
    if V16PDL_DEFAULT is not None:
        st["v16pdl_edges"] = {T: frozenset(V16PDL_DEFAULT) for T in V16PDL_T}
    plan = ((16, "sliding", True), (16, "full", False), (32, "sliding", False), (32, "full", True))
    for B, layer, stress in plan:
        T = 8 * B
        if not st["v16pdl_on"].get(T):
            continue
        try:
            data = _pdl8_data(B, 8, layer, gen, dev)
            off = dict(st, v16pdl_on={})
            on = dict(st, v16pdl_on={T: True})
            if V16PDL_DEFAULT is not None:
                on["v16pdl_edges"] = {T: frozenset(V16PDL_DEFAULT)}
            ids = _pdl8_flat(kernel_fn(data, off))[-1]
            if ids.dtype in (torch.int32, torch.int64) and ids.numel() == 8 * T:
                counts = torch.bincount(ids.flatten().long().clamp_min(0) // 32, minlength=8)
                data["local_start"].fill_(32 * int(counts.argmax()))
            kernel_fn(data, on)
            torch.cuda.synchronize()
            g_off, o_off = _pdl8_graph(lambda: kernel_fn(data, off))
            g_on, o_on = _pdl8_graph(lambda: kernel_fn(data, on))
        except Exception as e:
            st["v16pdl_on"][T] = False
            print(f"V16PDL T={T} off: {repr(e)[:400]}", flush=True)
            torch.cuda.synchronize()
            continue
        rows = _pdl8_time([g_off, g_on])
        d = sorted(r[0] - r[1] for r in rows)
        moff = sorted(r[0] for r in rows)[len(rows) // 2]
        mon = sorted(r[1] for r in rows)[len(rows) // 2]
        print(f"V16_AB T={T} {layer}: off {moff:.2f} on {mon:.2f} us, median paired saving {d[len(d) // 2]:.2f}, "
              f"wins {sum(1 for v in d if v > 0)}/{len(d)}", flush=True)
        if stress:
            g_off.replay()
            torch.cuda.synchronize()
            ref = [t.clone() for t in o_off]
            off_max, off_bad = _pdl8_drift(g_off, o_off, ref, PDL8_STRESS)
            on_max, on_bad = _pdl8_drift(g_on, o_on, ref, PDL8_STRESS)
            if not _pdl8h_guard(st, T, layer, off_max, off_bad, on_max, on_bad):
                st["v16pdl_on"][T] = False
        del g_off, g_on, o_off, o_on, data
        torch.cuda.empty_cache()
    print(f"V16PDL enabled {st['v16pdl_on']} edges {V16PDL_DEFAULT or V16PDL_EDGES} "
          f"(pre-wait boundary weights {'on' if V16PRE else 'off'})", flush=True)


_v16_prior_prepare = prepare
def prepare(weights):
    st = _v16_prior_prepare(weights)
    st["v16pdl_on"] = {}
    try:
        _v16_setup(st)
    except Exception as e:
        print("V16PDL off:", repr(e)[:400], flush=True)
        st["v16pdl_on"] = {}
        torch.cuda.synchronize()
    return st
'''


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    for old, new, n in EDITS:
        if text.count(old) != n:
            sys.exit(f"edit anchor found {text.count(old)} times, expected {n}: {old[:80]!r}")
        text = text.replace(old, new)
    name = DST.stem
    text = (f"# {name}: ly_stk1 with the T = 128 / 256 edges into the attention, row boundaries and route as PDL "
            f"secondaries{' and the boundary weights before the wait' if PRE else ''} (tools/dl_ly_v16pdl_patch.py).\n"
            + text + TAIL.replace("__V16PRE__", "True" if PRE else "False")
            .replace("__V16DEF__", repr(tuple(EDGES)) if EDGES else "None"))
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
