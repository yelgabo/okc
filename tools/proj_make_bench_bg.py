"""Write the batched-GEMM diagnostic bench (round 6): how much of the m = 256 qkv_gate and o
cuBLASLt time is per-launch ramp and tail, and how much is per-instance steady state.

    python3 tools/proj_make_bench_bg.py OUT.py

OUT.py is muse_bf16_proj/submission.py (the judged path, unchanged) plus a block that runs at
import, before the judged prepare(). It builds a copy of the production cuBLASLt host code
(LT_CPP) with one addition, lt_set_batch(b): plans created after it carry b strided-batched
instances on every operand (W, x and out each advance by one whole matrix per instance).

Per layer at m = 256, with B = 4 independent weight copies (reference.py's formula, separate
seeds) and 4 independent x / out buffers:

  b1      the production preference (LT_PREF, fixed algorithm and config), instance 0
  b1n     a second, separately created batch-1 plan on instance 1: the independent null
  b2 b4   the same fixed config as one strided-batched launch of 2 / 4 instances
  s2 s4   2 / 4 batch-1 launches back to back in one graph (instances 0..b-1)
  h2 h4   cuBLASLt's own top heuristic for the batched plan (config printed)
  M512 M1024  one GEMM with 512 / 1024 tokens on instance 0's weights (weight reuse: an upper
          reference, not a causal split)

Every arm is checked against an FP64 reference on sampled rows of every instance it writes.
Timing: cold-L2 graph replays (a 256 MB read before each), 60 rounds in a seeded random order,
the graph replay overhead measured and removed. The fit t(b) = a + s * b over b1 / b2 / b4
gives a (per-launch fixed: launch, ramp, wave tail) and s (per-instance steady state); s2 / s4
against b2 / b4 shows what separate launches add. Kernel names and device times of one replay
per arm come from torch.profiler when it is available. Lines:

  BG <layer> cfg <arm> ...   config fields, workspace, waves of each plan
  BG <layer> med ...         median us per arm after the replay overhead
  BG <layer> d b1 ...        paired median difference arm - b1 (us) / rounds faster
  BG <layer> fit ...         a, s, per-instance times, the null
  BG <layer> prof <arm> ...  kernels and their device us (one replay)
  BGCHECK                    dropped arms and why
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUND = "bg-a"

BENCH = r'''

_BG_ROUND = @@ROUND@@
_BG_LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024)}
_BG_M = 256
_BG_B = 4
import os as _bgos
import random as _bgrandom
import math as _bgmath

_BG_OUT = []


def _bgprint(s):
    print(s, flush=True)
    _BG_OUT.append(s)


def _bg_lt_src():
    src = LT_CPP
    anchors = ("cublasLtHandle_t g_handle = nullptr;\n", "int64_t lt_version() {\n",
               "    if (!p->la || !p->lb || !p->ld) return -4;\n")
    for a in anchors:
        if src.count(a) != 1:
            raise RuntimeError(f"LT_CPP anchor moved: {a.strip()}")
    src = src.replace(anchors[0], anchors[0] + "int g_batch = 1;\n")
    src = src.replace(anchors[1], "int64_t lt_set_batch(int64_t b) {\n    g_batch = (int)b;\n    return 0;\n}\n\n"
                      + anchors[1])
    batch = """    if (g_batch > 1) {
        const int32_t bc = g_batch;
        const int64_t sw = (int64_t)n * k, sx = (int64_t)m * k, so = (int64_t)m * n;
        const int64_t strides[3] = {row ? sx : sw, row ? sw : sx, so};
        cublasLtMatrixLayout_t ls[3] = {p->la, p->lb, p->ld};
        for (int i = 0; i < 3; ++i) {
            if (!ls[i]
                || cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc))
                       != CUBLAS_STATUS_SUCCESS
                || cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strides[i],
                                                    sizeof(strides[i])) != CUBLAS_STATUS_SUCCESS) {
                return -6;
            }
        }
    }
"""
    return src.replace(anchors[2], batch + anchors[2])


def _bg_build():
    src = _bg_lt_src()
    errs = []
    for i, flag in enumerate(("-l:libcublasLt.so.13", "-lcublasLt", "-l:libcublasLt.so.12")):
        try:
            return load_inline(name=f"okc_ltb_{i}", cpp_sources=src, functions=LT_FUNCS + ["lt_set_batch"],
                               with_cuda=True, extra_cflags=["-O2", "-std=c++20"], extra_ldflags=[flag],
                               verbose=False)
        except Exception as exc:
            errs.append(f"{flag}: {str(exc)[-200:]}")
    raise RuntimeError(" || ".join(errs))


def _bg_graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    return g


def _bg_time(graphs, rounds, flush, seed):
    rng = _bgrandom.Random(seed)
    names = list(graphs)
    samples = {n: [] for n in names}
    for _ in range(rounds):
        order = names[:]
        rng.shuffle(order)
        evs = []
        for n in order:
            flush.sum()
            a = torch.cuda.Event(enable_timing=True)
            z = torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[n].replay()
            z.record()
            evs.append((n, a, z))
        torch.cuda.synchronize()
        for n, a, z in evs:
            samples[n].append(1000.0 * a.elapsed_time(z))
    return samples


def _bg_plan(lt, m, n, k, layout, batch, cfg, ws_bytes):
    """(plan, idx, how) for the fixed config, else cuBLASLt's top heuristic; idx -1 if neither."""
    lt.lt_set_batch(batch)
    try:
        p = lt.lt_plan_new(m, n, k, layout, ws_bytes)
    finally:
        lt.lt_set_batch(1)
    if p < 0:
        return p, -1, f"plan {p}"
    if cfg is not None:
        idx = lt.lt_add_config(p, list(cfg))
        return p, idx, "fixed" if idx >= 0 else "fixed rejected"
    got = lt.lt_heuristic(p, ws_bytes, 8)
    return p, (0 if got > 0 else -1), f"heuristic {got}"


def _bg_layer(lt, layer, n, k, flush, ovh, dev, gen):
    m, B = _BG_M, _BG_B
    layout, cfg = LT_PREF[(layer, m)]
    ws = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)
    w = torch.stack([(torch.randn((n, k), generator=gen, device=dev) / _bgmath.sqrt(k)).to(torch.bfloat16)
                     for _ in range(B)]).contiguous()
    warg = w.transpose(1, 2).contiguous() if layout % 2 else w
    x = torch.randn((B, 4 * m, k), generator=gen, device=dev).to(torch.bfloat16).contiguous()
    xs = x[:, :m].contiguous()
    out = torch.empty((B, m, n), dtype=torch.bfloat16, device=dev)
    outM = torch.empty((4 * m, n), dtype=torch.bfloat16, device=dev)
    arms, checks, bad = {}, {}, []

    def add(name, mm, batch, use_cfg, wi, xt, ot):
        p, idx, how = _bg_plan(lt, mm, n, k, layout, batch, cfg if use_cfg else None, ws.numel())
        if idx < 0:
            bad.append(f"{layer}:{name}={how}")
            return None
        info = list(lt.lt_info(p, idx))
        if "@" not in name:
            _bgprint(f"BG {layer} cfg {name} L{layout} {_lt_cfg_str(info)} ({how})")

        def run(p=p, idx=idx, wt=warg[wi], xt=xt, ot=ot):
            rc = lt.lt_run(p, idx, wt, xt, ot, ws)
            if rc:
                raise RuntimeError(f"cublasLt status {rc}")
        return run

    single = {}
    for i in range(B):
        single[i] = add(f"b1@{i}" if i else "b1", m, 1, True, i, xs[i], out[i])
    if single[0] is not None:
        arms["b1"] = single[0]
        checks["b1"] = [(0, xs[0], out[0])]
    alt = add("b1n", m, 1, True, 1, xs[1], out[1])
    if alt is not None:
        arms["b1n"] = alt
        checks["b1n"] = [(1, xs[1], out[1])]
    for b in (2, 4):
        f = add(f"b{b}", m, b, True, 0, xs[:b], out[:b])
        if f is not None:
            arms[f"b{b}"] = f
            checks[f"b{b}"] = [(i, xs[i], out[i]) for i in range(b)]
        h = add(f"h{b}", m, b, False, 0, xs[:b], out[:b])
        if h is not None:
            arms[f"h{b}"] = h
            checks[f"h{b}"] = [(i, xs[i], out[i]) for i in range(b)]
        if all(single[i] is not None for i in range(b)):
            def seq(b=b):
                for i in range(b):
                    single[i]()
            arms[f"s{b}"] = seq
            checks[f"s{b}"] = [(i, xs[i], out[i]) for i in range(b)]
    for mm in (2 * m, 4 * m):
        f = add(f"M{mm}", mm, 1, True, 0, x[0, :mm], outM[:mm])
        if f is None:
            f = add(f"M{mm}h", mm, 1, False, 0, x[0, :mm], outM[:mm])
        if f is not None:
            arms[f"M{mm}"] = f
            checks[f"M{mm}"] = [(0, x[0, :mm], outM[:mm])]

    rows = torch.arange(0, m, 17, device=dev)
    for nm in list(arms):
        try:
            out.fill_(float("nan"))
            outM.fill_(float("nan"))
            arms[nm]()
            torch.cuda.synchronize()
            for i, xt, ot in checks[nm]:
                r = rows if xt.shape[0] == m else torch.arange(0, xt.shape[0], 17, device=dev)
                ref = xt[r].double() @ w[i].double().T
                err = float((ot[r].double() - ref).norm() / ref.norm())
                if not err < 5e-3:
                    raise RuntimeError(f"instance {i} rel err {err:.1e}")
        except Exception as exc:
            torch.cuda.synchronize()
            bad.append(f"{layer}:{nm}={str(exc)[-80:]}")
            arms.pop(nm)
    if bad:
        _bgprint("BGCHECK dropped: " + " ".join(bad))
    if "b1" not in arms:
        _bgprint(f"BG {layer}: b1 dropped, no timing")
        return
    graphs = {nm: _bg_graph(fn) for nm, fn in arms.items()}
    smp = _bg_time(graphs, 60, flush, 5000 + len(layer))
    med = {nm: statistics.median(v) - ovh for nm, v in smp.items()}
    dl = {nm: statistics.median([a - b for a, b in zip(v, smp["b1"])]) for nm, v in smp.items()}
    wins = {nm: sum(a < b for a, b in zip(v, smp["b1"])) for nm, v in smp.items()}
    order = sorted(med, key=med.get)
    _bgprint(f"BG {layer} med " + " ".join(f"{nm}={med[nm]:.2f}" for nm in order))
    _bgprint(f"BG {layer} d b1 " + " ".join(f"{nm}={dl[nm]:+.2f}/{wins[nm]}" for nm in order if nm != "b1"))
    pts = [(b, med[f"b{b}"]) for b in (1, 2, 4) if f"b{b}" in med]
    if len(pts) >= 2:
        mb = sum(b for b, _ in pts) / len(pts)
        mt = sum(t for _, t in pts) / len(pts)
        s = sum((b - mb) * (t - mt) for b, t in pts) / sum((b - mb) ** 2 for b, _ in pts)
        a = mt - s * mb
        per = " ".join(f"b{b}/inst={t / b:.2f}" for b, t in pts)
        seq = " ".join(f"s{b}/inst={med[f's{b}'] / b:.2f}" for b in (2, 4) if f"s{b}" in med)
        null = f"{dl['b1n']:+.2f}" if "b1n" in dl else "n/a"
        flops = 2.0 * m * n * k
        _bgprint(f"BG {layer} fit a={a:.2f} s={s:.2f} a/t1={a / med['b1']:.2f} {per} {seq} null(b1n-b1)={null} "
                 f"steady TFLOP/s {flops / (s * 1e-6) / 1e12:.0f} single {flops / (med['b1'] * 1e-6) / 1e12:.0f}")
    try:
        from torch.profiler import profile, ProfilerActivity
        for nm in order:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                flush.sum()
                graphs[nm].replay()
                torch.cuda.synchronize()
            ks = [(e.name, e.device_time_total) for e in prof.key_averages() if e.device_time_total > 0
                  and "reduce_kernel" not in e.name and "elementwise" not in e.name]
            _bgprint(f"BG {layer} prof {nm} " + " | ".join(f"{k_[:60]} {t:.1f}" for k_, t in ks[:4]))
    except Exception as exc:
        _bgprint(f"BG {layer} prof unavailable: {type(exc).__name__}: {str(exc)[-120:]}")
    graphs = None
    torch.cuda.empty_cache()


def _bg_run():
    dev = torch.device("cuda", torch.cuda.current_device())
    t0 = time.time()
    lt = _bg_build()
    _bgprint(f"BGBUILD {time.time() - t0:.1f} s, cublasLt {lt.lt_version()}, round {_BG_ROUND}")
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    tiny = torch.zeros(1, device=dev)
    one = _bg_graph(lambda: tiny.add_(1))

    def two_fn():
        tiny.add_(1)
        tiny.add_(1)
    two = _bg_graph(two_fn)
    ov = _bg_time({"one": one, "two": two}, 200, flush, 1)
    ovh = 2 * statistics.median(ov["one"]) - statistics.median(ov["two"])
    _bgprint(f"BGOVH {ovh:.2f} us")
    gen = torch.Generator(device=dev).manual_seed(4545)
    for layer, (n, k) in _BG_LAYERS.items():
        try:
            _bg_layer(lt, layer, n, k, flush, ovh, dev, gen)
        except Exception as exc:
            import traceback as _tb
            _bgprint(f"BG {layer} failed: {type(exc).__name__}: {str(exc)[:200]} {_tb.format_exc()[-600:]}")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    _bgprint(f"BGDONE {time.time() - t0:.1f} s")


if not _bgos.environ.get("PROJ_BG_DONE"):
    _bgos.environ["PROJ_BG_DONE"] = "1"
    try:
        _bg_run()
    except Exception as _exc:
        import traceback as _tb
        _bgprint(f"BGBENCH failed: {type(_exc).__name__}: {str(_exc)[:300]} {_tb.format_exc()[-900:]}")
    _bg_prepare_judged = prepare

    def prepare(weights):
        import contextlib as _ctx
        import io as _io
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            state = _bg_prepare_judged(weights)
        for ln in buf.getvalue().splitlines():
            if ln.startswith(("ltune", "prepare done", "lt build")) or "disabled" in ln:
                print("judged| " + ln, flush=True)
        for line in _BG_OUT:
            print(line, flush=True)
        return state
'''


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = Path(args.pop(0))
    sub = (ROOT / "muse_bf16_proj/submission.py").read_text()
    out.write_text(sub + BENCH.replace("@@ROUND@@", repr(ROUND)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
