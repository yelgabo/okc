"""Write the qkv_gate one-wave bench: the 4-CTA cluster K split (qkv8c4 / qkv16c4 / qkv16c4w /
qkv32c4 ...) against the current 96-CTA picks at m = 8 / 16 / 32, with independent duplicate
nulls.

    python3 tools/proj_make_bench_c4.py OUT.py

OUT.py is muse_bf16_proj/submission.py (the judged path, unchanged, which also offers the c4
configs to its own tuner) plus a block that runs at import, before the judged prepare(). The
block builds the production extension (A, cached for the judged prepare) and a second copy
under another name (B), and packs the judge-like qkv_gate weight separately for each. Every
arm passes the FP64 self-test, the c4 arms also the 2-ulp match against the unsplit pick
(one-hot rows at the K-split boundaries included), then a CUDA-graph replay check with fresh
inputs copied into the captured buffer twice, else it is dropped. Timing: cold-L2 graph
replays, ROUNDS rounds in a seeded random order. Lines (printed again at the end of prepare):

  C4 m=<m> med          median us per arm (graph replay, overhead not removed)
  C4 m=<m> d <ref>      paired median difference arm - ref, us / rounds the arm was faster
  C4CHECK               dropped arms and why
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUND = "c4-a"

BLOCK = r'''

# ---- bench block (tools/proj_make_bench_c4.py, round @@ROUND@@) ----
import os as _c4os
import random as _c4rand

_C4_OUT = []
_C4_ROUNDS = 100
_C4_MS = (8, 16, 32)
_C4_ARMS = {8: ["qkv8", "qkv8c4", "qkv8c4w"], 16: ["qkv16", "qkv16c4", "qkv16c4w"], 32: ["qkv32", "qkv32c4"]}


def _c4_print(s):
    print(s, flush=True)
    _C4_OUT.append(s)


def _c4_ext(tag):
    if tag == "A":
        return _build_ext()
    src = sk_source() + GEMV_CUDA_SRC + wd_source() + rt_source()
    kwargs = dict(name="okc_sk_v2_c4" + tag,
                  cpp_sources="#include <torch/extension.h>\n" + _SK_CPP + GEMV_CPP_DECLS + _WD_CPP + _RT_CPP,
                  cuda_sources=src,
                  functions=["sk_fwd", "skf_fwd", "cgemv_fwd", "rgemv_fwd", "sgemv_fwd", "wd_fwd", "rt_fwd"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    try:
        return load_inline(no_implicit_headers=True, **kwargs)
    except TypeError:
        kwargs["cuda_sources"] = src.replace("#include <ATen/core/Tensor.h>", "#include <torch/types.h>")
        return load_inline(**kwargs)


def _c4_replay_ok(fn, w, m, k, dev, gen):
    xs = torch.zeros((m, k), dtype=torch.bfloat16, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            out = fn(xs)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn(xs)
    for _ in range(2):
        x, probes = _gen_x(m, k, gen, dev)
        xs.copy_(x)
        g.replay()
        torch.cuda.synchronize()
        ref = (x.double() @ w.double().T).to(torch.bfloat16).double()
        ok, err, perr = _check(out, ref, probes, SELFTEST_TOL["bf16"])
        if not ok:
            return f"replay err {err:.2e} probe {perr:.2e}"
    return None


def _c4_run():
    dev = torch.device("cuda", torch.cuda.current_device())
    t0 = time.time()
    wgen = torch.Generator(device=dev).manual_seed(5151)
    w = (torch.randn((2304, 8192), generator=wgen, device=dev) / 90.50966799187809).to(torch.bfloat16)
    n, k = w.shape
    names = sorted({nm for lst in _C4_ARMS.values() for nm in lst})
    geoms = {nm: _sk_geom(nm) for nm in names}
    slices = {nm: n for nm in names}
    gen = torch.Generator(device=dev).manual_seed(4242)
    arms = {m: {} for m in _C4_MS}
    for tag in ("A", "B"):
        try:
            ext = _c4_ext(tag)
        except Exception as exc:
            _c4_print(f"C4CHECK ext {tag} FAIL {type(exc).__name__}: {str(exc)[-300:]}")
            continue
        packed = pack_layer(w, geoms, slices, balance=True)
        diag = torch.zeros(4, dtype=torch.int32, device=dev)
        if tag == "A":
            _c4_print(f"C4 pack base {packed['base']} escapes {packed['escapes']} pmax "
                      + " ".join(f"R{r}:{p}" for (r, _), (_, p) in packed["tabs"].items())
                      + f" build+pack {time.time() - t0:.1f} s")
        for m in _C4_MS:
            twin = None
            for nm in _C4_ARMS[m]:
                tab, pmax = packed["tabs"][(geoms[nm]["R"], n)]
                fn = _sk_runner(ext, nm, packed["streams"][nm], tab, pmax, packed["base4"], n, k, torch.bfloat16,
                                diag)
                ms = {8: (1, 5, 8), 16: (9, 16), 32: (17, 32)}[m]
                try:
                    _self_test(f"{tag}.{nm}", fn, w, n, ms, SELFTEST_TOL["bf16"], gen, diag)
                    if twin is None:
                        twin = fn
                    elif geoms[nm]["cl"] > 1:
                        cols = _escape_cols(tab, geoms[nm]["cl"], k)
                        _match_twin(fn, twin, ms, k, gen, dev, cols)
                        if tag == "A":
                            _c4_print(f"C4 twin {nm} m={m}: {len(cols)} one-hot columns identical")
                    bad = _c4_replay_ok(fn, w, m, k, dev, gen)
                    if bad:
                        raise RuntimeError(bad)
                except Exception as exc:
                    _c4_print(f"C4CHECK drop {tag}.{nm} m={m}: {type(exc).__name__}: {str(exc)[-200:]}")
                    continue
                arms[m][f"{tag}.{nm}"] = fn
    _c4_print(f"C4 checks done {time.time() - t0:.1f} s")
    flush = torch.ones(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    rng = _c4rand.Random(2026)
    xgen = torch.Generator(device=dev).manual_seed(777)
    for m in _C4_MS:
        live = dict(arms[m])
        live["sgl"] = lambda x: _cublas(x, w)
        x = torch.randn((m, k), generator=xgen, device=dev).to(torch.bfloat16)
        graphs = {}
        for nm, fn in live.items():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    fn(x)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                fn(x)
            graphs[nm] = g
        smp = {nm: [] for nm in graphs}
        for _ in range(_C4_ROUNDS):
            order = list(graphs)
            rng.shuffle(order)
            evs = []
            for nm in order:
                torch.sum(flush, out=sink)
                a = torch.cuda.Event(enable_timing=True)
                z = torch.cuda.Event(enable_timing=True)
                a.record()
                graphs[nm].replay()
                z.record()
                evs.append((nm, a, z))
            torch.cuda.synchronize()
            for nm, a, z in evs:
                smp[nm].append(1000.0 * a.elapsed_time(z))
        _c4_print(f"C4 m={m} med " + " ".join(f"{nm}={statistics.median(v):.2f}" for nm, v in smp.items()))
        for ref in (f"A.{_C4_ARMS[m][0]}",):
            if ref not in smp:
                continue
            parts = []
            for nm in smp:
                if nm == ref:
                    continue
                d = [u - v for u, v in zip(smp[nm], smp[ref])]
                parts.append(f"{nm}={statistics.median(d):+.2f}/{sum(1 for e in d if e < 0)}")
            _c4_print(f"C4 m={m} d {ref} " + " ".join(parts))
        pairs = [(f"A.{nm}", f"B.{_C4_ARMS[m][0]}") for nm in _C4_ARMS[m][1:]]
        pairs += [(f"B.{nm}", f"A.{nm}") for nm in _C4_ARMS[m][1:]]
        for a, b in pairs:
            if a in smp and b in smp:
                d = [u - v for u, v in zip(smp[a], smp[b])]
                _c4_print(f"C4 m={m} d {b} {a}={statistics.median(d):+.2f}/{sum(1 for e in d if e < 0)}")
        graphs = None
        torch.cuda.empty_cache()
    _c4_print(f"C4 done {time.time() - t0:.1f} s")


if not _c4os.environ.get("PROJ_C4_DONE"):
    _c4os.environ["PROJ_C4_DONE"] = "1"
    try:
        _c4_run()
    except Exception as _exc:
        _c4_print(f"C4CHECK bench failed: {type(_exc).__name__}: {str(_exc)[:400]}")
    _c4_prepare = prepare

    def prepare(weights):
        state = _c4_prepare(weights)
        for line in _C4_OUT:
            print(line, flush=True)
        return state
'''


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    base = (ROOT / "muse_bf16_proj/submission.py").read_text()
    out.write_text(base + BLOCK.replace("@@ROUND@@", ROUND))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
