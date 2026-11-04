#!/usr/bin/env python3
"""Route -> MoE PDL edge: writes muse_decode_layer/cand/ly_mpdl.py from ly_stk4.py (a054c8d5, checked).

Both MoE sources (MOE_CUDA_SRC for B4 / main, the helper0 alt source for TINY / TINY2 / MID) launch with
cudaLaunchKernelExC carrying cudaLaunchAttributeCooperative and cudaLaunchAttributeProgrammaticStreamSerialization
when okc_set_pdl(1) was called, and keep cudaLaunchCooperativeKernel otherwise. MOE_HAS_PDL becomes True; every
existing caller still passes _pdl(T), which is always False (PDL = False), so only the two PDL8 paths below can
turn the edge on. The kernel is unchanged: griddepcontrol.wait stays its first statement, every fence, barrier and
counter is untouched.

The MoE call sites inside _pdl8_sl_tile_after_attention and _pdl8_graft_after_attention (only those two of the
five identical blocks) set the edge when the PDL8
framework is on at T, T <= 32 and state["moe_pdl"][T] is true, and reset it to 0 after the launch, because
g_okc_pdl is a per-module static that persists across calls.

prepare runs a paired guard after the PDL8 setup (MPDL_GUARD lines): per T in 1 / 8 / 16 / 32 and both layers, it
captures an edge-off and an edge-on graph on one data set, then for MPDL_ROUNDS rounds writes a fresh random
residual stream (so routing changes) and a rotated local_start, and replays off, off, on, on. Off-vs-off measures
the layer's own nondeterminism; if that is bitwise zero the on replays must be bitwise identical to the first off
replay, otherwise the PDL8H criteria apply. A replay that makes no progress for MPDL_HANG_S seconds ends the
process. A T that fails, or raises, runs without the edge.

    python3 tools/dl_ly_mpdl_patch.py [--out muse_decode_layer/cand/ly_mpdl.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk4.py"
DST = ROOT / "muse_decode_layer/cand/ly_mpdl.py"
PIN = "a054c8d57555352d924bc2c4ae73f3199f05025f40a11300bc301d1655ba927e"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

LAUNCH_OLD = """  const cudaError_t launch=cudaLaunchCooperativeKernel((const void*)kernel,dim3((unsigned)grid),
      dim3(okc::NTL),args,sm,st);
"""
LAUNCH_NEW = """  cudaError_t launch;
  if(g_okc_pdl){
    cudaLaunchConfig_t lc={};
    lc.gridDim=dim3((unsigned)grid); lc.blockDim=dim3(okc::NTL); lc.dynamicSmemBytes=sm; lc.stream=st;
    cudaLaunchAttribute la[2];
    la[0].id=cudaLaunchAttributeCooperative; la[0].val.cooperative=1;
    la[1].id=cudaLaunchAttributeProgrammaticStreamSerialization; la[1].val.programmaticStreamSerializationAllowed=1;
    lc.attrs=la; lc.numAttrs=2;
    launch=cudaLaunchKernelExC(&lc,(const void*)kernel,args);
  } else {
    launch=cudaLaunchCooperativeKernel((const void*)kernel,dim3((unsigned)grid),dim3(okc::NTL),args,sm,st);
  }
"""

HAS_OLD = "MOE_HAS_PDL = False   # their extension exports okc_set_pdl (programmatic stream serialization)\n"
HAS_NEW = "MOE_HAS_PDL = True   # their extension exports okc_set_pdl (programmatic stream serialization)\n"

SITE_OLD = """        ext = _moe(T, state)
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
"""
SITE_NEW = """        ext = _moe(T, state)
        mpdl = _mpdl_on(T, state)
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if mpdl else 0)
        for m0 in range(0, T, _MA_MMAX):
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])
        if mpdl:
            ext.okc_set_pdl(0)
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
"""

HELPER_ANCHOR = "def _pdl8_edge(name):\n"
HELPER = '''def _mpdl_on(T, state):
    return MOE_HAS_PDL and _PDL8 and T <= 32 and bool(state.get("moe_pdl", {}).get(T, False))


'''

GUARD = r'''


MPDL_T = {1: (1, 1), 8: (1, 8), 16: (16, 1), 32: (32, 1)}
MPDL_ROUNDS = 48
MPDL_HANG_S = 30.0


def _mpdl_wait(what):
    import time as _mpdl_time
    ev = torch.cuda.Event()
    ev.record()
    t0 = _mpdl_time.time()
    while not ev.query():
        if _mpdl_time.time() - t0 > MPDL_HANG_S:
            print(f"MPDL_GUARD HANG during {what}: no progress in {MPDL_HANG_S:.0f} s, exiting", flush=True)
            os._exit(17)
        _mpdl_time.sleep(1e-4)


def _mpdl_cmp(outs, ref, acc):
    for a, b in zip(outs, ref):
        acc["bits"] += (a.reshape(-1).view(torch.uint8) != b.reshape(-1).view(torch.uint8)).sum()
        if a.is_floating_point():
            d = (a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)
            acc["rel"] = torch.maximum(acc["rel"], torch.nan_to_num(d, nan=1e30))
        else:
            acc["ints"] += (a != b).sum()


def _mpdl_layer(st, T, layer, gen, dev):
    B, q = MPDL_T[T]
    data = _pdl8_data(B, q, layer, gen, dev, ctx=8192)
    off = dict(st, moe_pdl={})
    on = dict(st, moe_pdl={T: True})
    kernel_fn(data, off)
    kernel_fn(data, on)
    _mpdl_wait(f"T={T} {layer} eager")
    g_off, o_off = _pdl8_graph(lambda: kernel_fn(data, off))
    g_on, o_on = _pdl8_graph(lambda: kernel_fn(data, on))
    z = lambda: {"bits": torch.zeros((), dtype=torch.int64, device=dev), "ints": torch.zeros((), dtype=torch.int64, device=dev),
                 "rel": torch.zeros((), device=dev)}
    a_off, a_on = z(), z()
    s_buf, x_buf = data["stream"], data["attn_input"]
    for r in range(MPDL_ROUNDS):
        s = torch.randn(s_buf.shape, generator=gen, device=dev, dtype=s_buf.dtype)
        x = (s.float() * torch.rsqrt(s.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(x_buf.dtype)
        data["local_start"].fill_(32 * (r % 8))
        ref = None
        for g, o, acc in ((g_off, o_off, None), (g_off, o_off, a_off), (g_on, o_on, a_on), (g_on, o_on, a_on)):
            s_buf.copy_(s)
            x_buf.copy_(x)
            g.replay()
            if ref is None:
                ref = [t.clone() for t in o]
            else:
                _mpdl_cmp(o, ref, acc)
        _mpdl_wait(f"T={T} {layer} round {r}")
    res = {k: (float(v) if k == "rel" else int(v)) for k, v in a_off.items()}, {k: (float(v) if k == "rel" else int(v)) for k, v in a_on.items()}
    del g_off, g_on, o_off, o_on, data
    torch.cuda.empty_cache()
    return res


def _mpdl_setup(st):
    import time as _mpdl_time
    t0 = _mpdl_time.time()
    st["moe_pdl"] = {}
    if not MOE_HAS_PDL:
        return
    dev = st["qkv"].device
    gen = torch.Generator(device=dev).manual_seed(27182)
    for T in MPDL_T:
        if not st.get("pdl8_on", {}).get(T):
            print(f"MPDL_GUARD T={T}: PDL8 off at this T, edge off", flush=True)
            continue
        ok = True
        for layer in ("sliding", "full"):
            try:
                off, on = _mpdl_layer(st, T, layer, gen, dev)
            except Exception as e:
                print(f"MPDL_GUARD T={T} {layer}: {repr(e)[:400]}, edge off", flush=True)
                torch.cuda.synchronize()
                ok = False
                break
            if off["bits"] == 0:
                good = on["bits"] == 0
                rule = "bitwise (off vs off is bitwise equal)"
            else:
                good = (on["ints"] <= PDL8H_IDS_MULT * off["ints"] + PDL8H_IDS_SLACK
                        and on["rel"] <= max(PDL8H_BOUND, off["rel"]))
                rule = f"PDL8H (ids <= {PDL8H_IDS_MULT}*off+{PDL8H_IDS_SLACK}, rel <= max({PDL8H_BOUND:.0e}, off))"
            print(f"MPDL_GUARD T={T} {layer}: {MPDL_ROUNDS} fresh-routing rounds, 2 replays per arm per round; "
                  f"off vs off bytes {off['bits']} ids {off['ints']} rel {off['rel']:.2e}; "
                  f"on vs off bytes {on['bits']} ids {on['ints']} rel {on['rel']:.2e}; rule {rule}: "
                  f"{'pass' if good else 'FAIL, edge off'}", flush=True)
            ok = ok and good
            if not good:
                break
        st["moe_pdl"][T] = ok
    print(f"MPDL_GUARD done in {_mpdl_time.time() - t0:.1f} s; moe_pdl {st['moe_pdl']}", flush=True)


_mpdl_prior_prepare = prepare


def prepare(weights):
    st = _mpdl_prior_prepare(weights)
    try:
        _mpdl_setup(st)
    except Exception as e:
        print("MPDL_GUARD off:", repr(e)[:400], flush=True)
        st["moe_pdl"] = {}
        torch.cuda.synchronize()
    return st
'''


def sub(s, old, new, n):
    if s.count(old) != n:
        raise SystemExit(f"anchor {old[:70]!r}: {s.count(old)} found, {n} expected")
    return s.replace(old, new)


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != PIN:
        raise SystemExit(f"{SRC.name} sha {sha} != pinned {PIN}")
    s = raw.decode()
    s = sub(s, LAUNCH_OLD, LAUNCH_NEW, 2)
    s = sub(s, HAS_OLD, HAS_NEW, 1)
    for fn in ("def _pdl8_sl_tile_after_attention(", "def _pdl8_graft_after_attention("):
        if s.count(fn) != 1:
            raise SystemExit(f"anchor {fn!r}: {s.count(fn)} found, 1 expected")
        a = s.index(fn)
        b = s.index(SITE_OLD, a)
        if s.find("\ndef ", a + 1) < b:
            raise SystemExit(f"MoE call site not inside {fn}")
        s = s[:b] + SITE_NEW + s[b + len(SITE_OLD):]
    s = sub(s, HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    s = s.rstrip("\n") + "\n" + GUARD
    DST.write_text(s)
    py_compile.compile(str(DST), doraise=True)
    print(f"{DST.relative_to(ROOT)} sha256 {hashlib.sha256(DST.read_bytes()).hexdigest()} from {SRC.name} {sha[:8]}")


if __name__ == "__main__":
    main()
