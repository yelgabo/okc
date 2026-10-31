"""Write the codec factorial bench for the m = 1 GEMV (cgemv) and the register-direct kernel (rd):
the check that decides pair7 again after the sk factorial (077b210c2087470d) left it at +0.85%.

    python3 tools/proj_make_bench_cx.py OUT.py

OUT.py is muse_bf16_proj/submission.py (the judged path, unchanged) plus a block that runs at
import, before the judged prepare(). It builds the production extension and a bench extension
compiled with OKC_BENCH_VARIANTS (cgemvx_fwd, rdx_fwd: the same kernels with the VB bits of
cgemv_body / rd_kernel). Arms per case: "prod" (production extension), and VB sets from the bench
extension: v0 (the null), v1 / v2 (bytes only, 736 / 704-byte tiles' worth, timing only), v4
(decode ALU), v12 (decode ALU + LUT loads), v13 (the pair7 model), v14 (the q12 model). Cases:
cgemv at m = 1 on qkv_gate, o, pooler and the unpooler slice; rd at pooler 8 (rdp8) and o 8
(rdo8). Exact arms are checked against FP64 with probe rows; timing-only arms are not. Timing:
cold-L2 graph replays, 100 rounds, seeded random order. Lines:

  CX <case> med      median us per arm after the replay overhead
  CX <case> d prod   paired median difference arm - prod (us) / rounds faster
  CXCHECK            dropped arms and why
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUND = "cx-a"
TIMING_ONLY = (1, 2, 13, 14)

BENCH = r'''

_CX_ROUND = @@ROUND@@
_CX_TONLY = @@TONLY@@
import os as _cxos
import random as _cxrandom
import math as _cxmath

_CX_OUT = []


def _cxprint(s):
    print(s, flush=True)
    _CX_OUT.append(s)


def _cx_build():
    cpp = ("#include <torch/extension.h>\n" + GEMV_CPP_DECLS + _SK_CPP + _RD_CPP
           + "void cgemvx_fwd(at::Tensor stream, at::Tensor table, int64_t pmax, int64_t base4, at::Tensor x, "
           "at::Tensor out, int64_t nthreads, int64_t vb);\n"
           "void rdx_fwd(int64_t cfg, at::Tensor ws, at::Tensor esc, at::Tensor x, at::Tensor out, int64_t n, "
           "int64_t k, int64_t base4, int64_t vb);\n")
    g = globals()
    saved = {nm: g[nm] for nm in ("SK_CONFIGS", "SK_INDEX")}
    try:
        g["SK_CONFIGS"] = []
        g["SK_INDEX"] = {}
        src = GEMV_CUDA_SRC + sk_source() + rd_source()
    finally:
        g.update(saved)
    return load_inline(name="okc_cx_bench", cpp_sources=cpp, cuda_sources=src,
                       functions=["cgemvx_fwd", "rdx_fwd"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DOKC_BENCH_VARIANTS"],
                       verbose=False, no_implicit_headers=True)


def _cx_graph(fn, xbuf):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(xbuf)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn(xbuf)
    torch.cuda.synchronize()
    return g, out


def _cx_time(graphs, rounds, flush, seed):
    rng = _cxrandom.Random(seed)
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


def _cx_case(label, arms, w, n, m, flush, ovh, gen, dev, seed):
    k = w.shape[1]
    wd = w[:n].double()
    bad = []
    x, probes = _gen_x(m, k, gen, dev)
    ref = x.double() @ wd.T
    for nm in list(arms):
        if nm.startswith("v") and int(nm[1:]) in _CX_TONLY:
            continue
        try:
            out = arms[nm](x)
            torch.cuda.synchronize()
            ok, err, perr = _check(out, ref, probes, SELFTEST_TOL["bf16"])
            if not ok:
                raise RuntimeError(f"err {err:.1e}/{perr:.1e}")
        except Exception as exc:
            torch.cuda.synchronize()
            bad.append(f"{nm}={str(exc)[-80:]}")
            arms.pop(nm)
    xbuf, _ = _gen_x(m, k, gen, dev)
    graphs = {}
    for nm in list(arms):
        try:
            g, _ = _cx_graph(arms[nm], xbuf)
            graphs[nm] = g
        except Exception as exc:
            torch.cuda.synchronize()
            bad.append(f"{nm}@graph={str(exc)[-80:]}")
    if bad:
        _cxprint(f"CXCHECK {label} dropped: " + " ".join(bad))
    if "prod" not in graphs:
        _cxprint(f"CX {label}: prod dropped")
        return
    smp = _cx_time(graphs, 100, flush, seed)
    med = {nm: statistics.median(v) - ovh for nm, v in smp.items()}
    dl = {nm: statistics.median([a - b for a, b in zip(v, smp["prod"])]) for nm, v in smp.items()}
    wins = {nm: sum(a < b for a, b in zip(v, smp["prod"])) for nm, v in smp.items()}
    order = sorted(med, key=med.get)
    _cxprint(f"CX {label} med " + " ".join(f"{nm}={med[nm]:.2f}" for nm in order))
    _cxprint(f"CX {label} d prod " + " ".join(f"{nm}={dl[nm]:+.2f}/{wins[nm]}" for nm in order if nm != "prod"))


def _cx_run():
    dev = torch.device("cuda", torch.cuda.current_device())
    t0 = time.time()
    prod = _build_ext()
    t1 = time.time()
    bench = _cx_build()
    _cxprint(f"CXBUILD prod {t1 - t0:.1f} s, bench {time.time() - t1:.1f} s, round {_CX_ROUND}, vb {list(CODEC_VB)}")
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    tiny = torch.zeros(1, device=dev)
    one, _ = _cx_graph(lambda x: tiny.add_(1), None)

    def two_fn(x):
        tiny.add_(1)
        tiny.add_(1)
    two, _ = _cx_graph(two_fn, None)
    ov = _cx_time({"one": one, "two": two}, 200, flush, 1)
    ovh = 2 * statistics.median(ov["one"]) - statistics.median(ov["two"])
    _cxprint(f"CXOVH {ovh:.2f} us")
    gen = torch.Generator(device=dev).manual_seed(4848)
    shapes = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
    weights = {nm: (torch.randn(s, generator=gen, device=dev) / _cxmath.sqrt(s[1])).to(torch.bfloat16).contiguous()
               for nm, s in shapes.items()}
    ci = 0
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        try:
            w = weights[layer]
            n = UNPOOL_SLICE if layer == "unpooler" else w.shape[0]
            w1 = w[:n].contiguous()
            cg = pack_cgemv(w1)
            if cg is None:
                _cxprint(f"CX cgemv {layer}: rejected")
                continue
            nth = cgemv_threads(*w1.shape)

            def mk(ext, vb, x_fwd):
                def run(x):
                    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
                    if x_fwd:
                        ext.cgemvx_fwd(cg["stream"], cg["table"], cg["pmax"], cg["base4"], x, out, nth, vb)
                    else:
                        ext.cgemv_fwd(cg["stream"], cg["table"], cg["pmax"], cg["base4"], x, out, nth)
                    return out
                return run
            arms = {"prod": mk(prod, 0, False)}
            for vb in CODEC_VB:
                arms[f"v{vb}"] = mk(bench, vb, True)
            ci += 1
            _cx_case(f"cgemv {layer} m=1", arms, w, n, 1, flush, ovh, gen, dev, 6000 + ci)
        except Exception as exc:
            import traceback as _tb
            _cxprint(f"CX cgemv {layer} failed: {type(exc).__name__}: {str(exc)[:200]} {_tb.format_exc()[-400:]}")
        torch.cuda.synchronize()
    for layer, rnm in (("pooler", "rdp8"), ("o", "rdo8")):
        try:
            w = weights[layer]
            n, k = w.shape
            pk = rd_pack(w, rnm)
            if pk is None:
                _cxprint(f"CX rd {rnm}: not packable")
                continue
            cid = RD_NAMES.index(rnm)

            def mkr(ext, vb, x_fwd, n=n, k=k, pk=pk, cid=cid):
                def run(x):
                    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
                    if x_fwd:
                        ext.rdx_fwd(cid, pk["stream"], pk["esc"], x, out, n, k, pk["base4"], vb)
                    else:
                        ext.rd_fwd(cid, pk["stream"], pk["esc"], x, out, n, k, pk["base4"])
                    return out
                return run
            arms = {"prod": mkr(prod, 0, False)}
            for vb in CODEC_VB:
                arms[f"v{vb}"] = mkr(bench, vb, True)
            ci += 1
            _cx_case(f"rd {rnm} m=8", arms, w, n, 8, flush, ovh, gen, dev, 6000 + ci)
        except Exception as exc:
            import traceback as _tb
            _cxprint(f"CX rd {rnm} failed: {type(exc).__name__}: {str(exc)[:200]} {_tb.format_exc()[-400:]}")
        torch.cuda.synchronize()
    _cxprint(f"CXDONE {time.time() - t0:.1f} s")
    torch.cuda.empty_cache()


if not _cxos.environ.get("PROJ_CX_DONE"):
    _cxos.environ["PROJ_CX_DONE"] = "1"
    try:
        _cx_run()
    except Exception as _exc:
        import traceback as _tb
        _cxprint(f"CXBENCH failed: {type(_exc).__name__}: {str(_exc)[:300]} {_tb.format_exc()[-900:]}")
    _cx_prepare_judged = prepare

    def prepare(weights):
        import contextlib as _ctx
        import io as _io
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            state = _cx_prepare_judged(weights)
        for ln in buf.getvalue().splitlines():
            if ln.startswith(("prepare done", "sk build")) or "disabled" in ln:
                print("judged| " + ln, flush=True)
        for line in _CX_OUT:
            print(line, flush=True)
        return state
'''


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = Path(args.pop(0))
    sub = (ROOT / "muse_bf16_proj/submission.py").read_text()
    out.write_text(sub + BENCH.replace("@@ROUND@@", repr(ROUND)).replace("@@TONLY@@", repr(TIMING_ONLY)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
