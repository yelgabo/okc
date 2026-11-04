"""Run an okc submission file on a local H200 the way the judge times it.

python tools/box_run.py <problem dir> <submission.py> [--baseline] [--judge J.json ...]
    [--loads "decode 1,verify 1"] [--runs N] [--no-check] [--out result.json]

Per shape: prepare(weights) once, a correctness check against the problem's reference.py on one
fresh input, then one CUDA graph captured on a static input set and replayed `runs` times. Each
replay gets a fresh input set copied into the graph's buffers, a cold L2 (a 1 GiB read), and is
timed with CUDA events. With --baseline, SGLang's baseline.py is captured the same way and the two
replays alternate order per repetition, so `paired` is the median of baseline/submission ratios.
The judge's fixed replay cost is 2 * t(1-kernel graph) - t(2-kernel graph), read from its
replay_overhead records (one_ns, two_ns, overhead_ns); it is measured and subtracted here the same
way. --judge takes judge JSONs (okc profile JSON with "benchmarks", or a job JSON with
result.records), reuses their case seeds and prints judge vs box per cell.
"""
import argparse
import importlib.util
import inspect
import json
import os
import statistics
import sys
import threading
import time

_T0 = time.monotonic()
import torch

PH = {}
PDL8 = {}
_LAST = [_T0]


def phase(name):
    now = time.monotonic()
    PH[name] = PH.get(name, 0.0) + now - _LAST[0]
    _LAST[0] = now

CASES = {
    "muse_decode_layer": [
        dict(batch=b, ctx=80000, dist="soft", layer=layer, load=load, page=1, weight=w,
             **({"q_len": 8} if q else {}), **({"spread": "agentic"} if b > 1 else {}))
        for load, b, q in [("decode 1", 1, 0), ("decode 16", 16, 0), ("decode 32", 32, 0),
                           ("verify 1", 1, 1), ("verify 16", 16, 1), ("verify 32", 32, 1)]
        for layer, w in (("sliding", 46), ("full", 16))
    ],
    "muse_attention_decode": [
        dict(batch=b, ctx=80000, heads="tp8", dist="soft" if layer == "sliding" else "coherent", layer=layer,
             load=load, page=1, weight=w, **({"q_len": 8} if q else {}), **({"spread": "agentic"} if b > 1 else {}))
        for load, b, q in [("decode 1", 1, 0), ("decode 16", 16, 0), ("decode 32", 32, 0),
                           ("verify 1", 1, 1), ("verify 16", 16, 1), ("verify 32", 32, 1)]
        for layer, w in (("sliding", 46), ("full", 16))
    ],
    "muse_moe_w4a16": [dict(bs=bs, sharding="ep8", load=load, weight=60) for load, bs in
                       [("decode 1", 1), ("decode 16", 16), ("decode 32", 32), ("verify 1", 8),
                        ("verify 16", 128), ("verify 32", 256)]],
    "muse_bf16_proj": [dict(m=m, layer=layer, load=load, weight=62) for load, m in
                       [("decode 1", 1), ("decode 16", 16), ("decode 32", 32), ("verify 1", 8),
                        ("verify 16", 128), ("verify 32", 256)]
                       for layer in ("qkv_gate", "o", "pooler", "unpooler", "router")],
}
IGNORED = {"seed", "repeats", "weight", "load"}
FLUSH_BYTES = 1 << 30


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cell_key(case):
    return tuple(sorted((k, str(v)) for k, v in case.items() if k not in IGNORED))


def read_judge(paths):
    """{cell key: [record, ...]} over every judge JSON given, plus the case dicts in order."""
    cells, order = {}, []
    for p in paths:
        d = json.load(open(p))
        recs = d.get("benchmarks") or d.get("result", {}).get("records", [])
        for r in recs:
            if r.get("event") != "benchmark":
                continue
            k = cell_key(r["case"])
            if k not in cells:
                order.append(r["case"])
            cells.setdefault(k, []).append(r)
    return cells, order


HW = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}


def implied_distinct(ref, record):
    """The router_distinct target a judge cell's speed_of_light_ns implies (bisection on the
    reference's own bound); None when the bound does not depend on it."""
    case, sol = record["case"], record.get("speed_of_light_ns")
    if sol is None or not hasattr(ref, "speed_of_light_ns"):
        return None
    f = lambda d: ref.speed_of_light_ns(**{**case, "router_distinct": d}, hw=HW)  # noqa: E731
    lo, hi = 0.0, 64.0
    if f(hi) < sol or f(lo) > sol * (1 + 1e-9) or abs(f(hi) - f(lo)) < 1e-6:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < sol else (lo, mid)
    return round((lo + hi) / 2, 3)


def distinct_router_seed(ref, weights, cases, targets, seed, attempts):
    """weights_fields restricted to the distinct-experts column: the judge also matches mean
    routes (router_load), which no public record exposes."""
    loads = {}
    for i, c in enumerate(cases):
        if c.get("load") in targets:
            loads.setdefault(c["load"], []).append(i)
    hs = []
    with torch.no_grad():
        for c in cases:
            n = ref.ROUTE_SAMPLES if c.get("batch", 1) * c.get("q_len", 1) <= ref.SAMPLE_TOKENS else 1
            seeds = [c["seed"]] + [(c["seed"] + j * 0x2545F4914F6CDD1D) % (2**63 - 1) for j in range(1, n)]
            hs.append((torch.cat([ref.ref_front(ref.generate_input(**{**c, "seed": s, "local_start": 0}), weights)["h"]
                                  for s in seeds]), n))
        best = (float("inf"), None)
        for i in range(1, attempts + 1):
            rs = (seed + i * 0x9E3779B97F4A7C15) % (2**63 - 1)
            router = ref._router(torch.Generator(device="cuda").manual_seed(rs), "cuda", weights["pre_ffn_norm"])
            got = {}
            for load, members in loads.items():
                w = [float(cases[m].get("weight", 1)) for m in members]
                vals = [float(ref._mean_busiest_load(hs[m][0], router, hs[m][1])[1]) for m in members]
                got[load] = sum(a * b for a, b in zip(w, vals)) / sum(w)
            err = max(abs(got[k] / targets[k] - 1) for k in got)
            if err < best[0]:
                best = (err, rs, got)
            if err <= ref.LOAD_TOL:
                break
    return best


class LocalStartCache:
    """case_fields' local_start per fresh input, on disk. It is a function of reference.py, the
    weights (seed, router seed), the case and the input seed, and it is 90% of a run's input
    building (ref_front's FP32 attention), so every run with the same router seed recomputes the
    same values (docs/box-audit.md, "Input generation")."""

    def __init__(self, path, ref_path, router_seed, weights_seed):
        import hashlib
        self.path = path
        self.tag = [hashlib.sha256(open(ref_path, "rb").read()).hexdigest()[:16], router_seed, weights_seed]
        self.data = self._read()
        self.hits = self.misses = 0

    def _read(self):
        try:
            with open(self.path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def get(self, ref, case, seed, weights):
        key = json.dumps(self.tag + [sorted((k, str(v)) for k, v in case.items() if k not in ("seed", "local_start")), seed])
        if key in self.data:
            self.hits += 1
            return {"local_start": self.data[key]}
        self.misses += 1
        extra = ref.case_fields({**case, "seed": seed}, weights)
        self.data[key] = int(extra["local_start"])
        return extra

    def save(self):
        if not self.misses:
            return
        merged = {**self._read(), **self.data}
        tmp = f"{self.path}.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(merged, f)
        os.replace(tmp, self.path)


class GpuLock:
    """gpu.lock held on an fd this process owns (flock(2), the same lock flock(1) and box_bg.sh
    take), so it can be let go while a compiler runs and taken back before the next GPU work."""

    def __init__(self, path):
        import fcntl
        self.fcntl = fcntl
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        self.held = False
        self.events = []

    def _log(self, what, why, waited=None):
        t = time.time()
        self.events.append([round(t, 3), what, why] + ([round(waited, 3)] if waited is not None else []))
        extra = f" after {waited:.1f} s" if waited is not None else ""
        print(f"BXLOCK {t:.3f} {what} {why}{extra}", flush=True)

    def acquire(self, why):
        if self.held:
            return
        t0 = time.time()
        self.fcntl.flock(self.fd, self.fcntl.LOCK_EX)
        self.held = True
        self._log("acquire", why, time.time() - t0)

    def release(self, why):
        if not self.held:
            return
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        self.fcntl.flock(self.fd, self.fcntl.LOCK_UN)
        self.held = False
        self._log("release", why)


LOCK = None
_BUILDS = threading.Lock()
_ACTIVE = [0]


def unlocked(why, fn, *args, **kw):
    """Run a compiler call with gpu.lock released; it is held again before the last concurrent
    build returns. g1b4_stage_release builds several extensions from threads at once, so the
    release and the reacquire are counted rather than paired per call."""
    if LOCK is None:
        return fn(*args, **kw)
    with _BUILDS:
        _ACTIVE[0] += 1
        if _ACTIVE[0] == 1 and LOCK.held:
            LOCK.release(why)
    print(f"BXBUILD {time.time():.3f} start {why}", flush=True)
    try:
        return fn(*args, **kw)
    finally:
        print(f"BXBUILD {time.time():.3f} end {why}", flush=True)
        with _BUILDS:
            _ACTIVE[0] -= 1
            if _ACTIVE[0] == 0 and not LOCK.held:
                LOCK.acquire(why + " done")


def outer_flock_holds(path):
    """True when a parent process is flock(1) on this lock file: taking it again here would wait forever."""
    pid, real = os.getppid(), os.path.realpath(path)
    while pid > 1:
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
        if argv and os.path.basename(argv[0].decode(errors="replace")) == "flock":
            if any(os.path.realpath(a.decode(errors="replace")) == real for a in argv[1:] if a):
                return True
    return False


def triton_cache_hit(src, target=None, options=None, _env_vars=None):
    """The cache lookup at the top of triton.compiler.compile (triton 3.8), without compiling.
    Anything unexpected counts as a hit, which keeps the lock (the safe side)."""
    import hashlib
    from triton.compiler import compiler as tc
    try:
        if not isinstance(src, tc.ASTSource) or tc.knobs.compilation.always_compile:
            return True
        if tc.knobs.runtime.add_stages_inspection_hook is not None:
            return True
        target = target or tc.driver.active.get_current_target()
        backend = tc.make_backend(target)
        opts = backend.parse_options(dict(options or dict(), **src.parse_options()))
        env = tc.get_cache_invalidating_env_vars() if _env_vars is None else _env_vars
        key = tc.get_cache_key(src, backend, opts, env_vars=env)
        name = f"{src.name[:150]}.json"
        group = tc.get_cache_manager(hashlib.sha256(key.encode("utf-8")).hexdigest()).get_group(name) or {}
        return group.get(name) is not None
    except Exception:
        return True


def release_lock_for_triton():
    import triton.compiler
    orig = triton.compiler.compile

    def compile(src, target=None, options=None, _env_vars=None):
        if LOCK is not None and LOCK.held and not triton_cache_hit(src, target, options, _env_vars):
            return unlocked(f"triton {getattr(src, 'name', '?')}", orig, src, target=target, options=options,
                            _env_vars=_env_vars)
        return orig(src, target=target, options=options, _env_vars=_env_vars)
    triton.compiler.compile = compile


def content_addressed_extensions(root):
    """load_inline builds into root/<name>-<hash of every build argument>. torch keys its cache by
    name only, so two files that compile different sources under one name rebuild on every switch
    (g1b4_stage_release and g1b4_rs both own muse_att_v298w1pf_90a, docs/box-audit.md)."""
    import hashlib
    import torch.utils.cpp_extension as ce
    orig = ce.load_inline
    sig = inspect.signature(orig)

    def load_inline(*args, **kw):
        b = sig.bind(*args, **kw)
        if b.arguments.get("build_directory") is None:
            key = {k: v for k, v in b.arguments.items() if k not in ("build_directory", "verbose")}
            key["arch"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
            h = hashlib.sha256(repr(sorted(key.items())).encode()).hexdigest()[:16]
            b.arguments["build_directory"] = os.path.join(root, f"{b.arguments['name']}-{h}")
            os.makedirs(b.arguments["build_directory"], exist_ok=True)
        d = b.arguments["build_directory"]
        # A .so in the hashed directory means these exact arguments were built before: loading it
        # is quick, so keep the lock rather than queue behind a whole job for it.
        if any(f.endswith(".so") for f in os.listdir(d)):
            return orig(*b.args, **b.kwargs)
        return unlocked(f"build {b.arguments['name']}", orig, *b.args, **b.kwargs)
    ce.load_inline = load_inline

    orig_load = ce.load
    sig_load = inspect.signature(orig_load)

    def load(*args, **kw):
        b = sig_load.bind(*args, **kw)
        if b.arguments.get("build_directory") is None:
            srcs = b.arguments["sources"]
            srcs = [srcs] if isinstance(srcs, str) else list(srcs)
            key = {k: v for k, v in b.arguments.items() if k not in ("build_directory", "verbose")}
            key["arch"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
            key["contents"] = [hashlib.sha256(open(f, "rb").read()).hexdigest() for f in srcs]
            h = hashlib.sha256(repr(sorted(key.items())).encode()).hexdigest()[:16]
            b.arguments["build_directory"] = os.path.join(root, f"{b.arguments['name']}-{h}")
            os.makedirs(b.arguments["build_directory"], exist_ok=True)
        d = b.arguments["build_directory"]
        if any(f.endswith(".so") for f in os.listdir(d)):
            return orig_load(*b.args, **b.kwargs)
        return unlocked(f"build {b.arguments['name']}", orig_load, *b.args, **b.kwargs)
    ce.load = load


def pdl8_state(state):
    if not isinstance(state, dict) or "pdl8_on" not in state:
        return None
    return dict(on={int(k): bool(v) for k, v in state["pdl8_on"].items()},
                edges={int(k): sorted(v) for k, v in (state.get("pdl8_edges") or {}).items()})


def call(fn, data, state):
    return fn(data, state) if state is not None else fn(data)


def tensors(data):
    return {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}


def copy_into(static, fresh):
    for k, v in tensors(static).items():
        f = fresh.get(k)
        if isinstance(f, torch.Tensor) and f.shape == v.shape and f.dtype == v.dtype:
            v.copy_(f, non_blocking=True)


def capture(fn, data, state):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            call(fn, data, state)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = call(fn, data, state)
    torch.cuda.synchronize()
    return g, out


class Timer:
    def __init__(self):
        self.flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda").uniform_()
        self.sink = torch.empty((), dtype=torch.float32, device="cuda")
        self.a = torch.cuda.Event(enable_timing=True)
        self.b = torch.cuda.Event(enable_timing=True)

    def cold(self, g):
        torch.sum(self.flush, dim=0, out=self.sink)
        self.a.record()
        g.replay()
        self.b.record()
        self.b.synchronize()
        return self.a.elapsed_time(self.b) * 1e6

    def overhead(self, n=200):
        x = torch.zeros(1, device="cuda")
        graphs = []
        for kernels in (1, 2):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(kernels):
                    x.add_(1)
            graphs.append(g)
        one = statistics.median(self.cold(graphs[0]) for _ in range(n))
        two = statistics.median(self.cold(graphs[1]) for _ in range(n))
        return dict(one_ns=one, two_ns=two, overhead_ns=2 * one - two)


def check(ref, data, out, weights):
    fn = ref.check_implementation
    nparams = len(inspect.signature(fn).parameters)
    if hasattr(ref, "collect"):
        raw = ref.collect(data, out)
        return fn(data, raw, weights) if nparams == 3 else fn(data, raw)
    return fn(data, out, weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("submission")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--loads", default="")
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--inputs", type=int, default=0, help="distinct fresh input sets cycled per shape (0: one per run)")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--router-seed", type=int, default=None, help="skip the search and use this router seed")
    ap.add_argument("--router-attempts", type=int, default=2000)
    ap.add_argument("--case-local-start", action="store_true",
                    help="reuse the case seed's local_start for every fresh input (the runner before 2026-11-03)")
    ap.add_argument("--kernels", action="store_true", help="also print a per-kernel GPU time table (torch.profiler, 10 cold replays)")
    ap.add_argument("--ls-cache", default="",
                    help="JSON cache of case_fields' local_start per fresh input, e.g. /root/okc/.build/box/ls_cache.json")
    ap.add_argument("--lock", default="",
                    help="take this flock file in-process (e.g. /root/okc/.build/box/gpu.lock) and let it go while an "
                         "extension or Triton kernel compiles; implies --ext-cache /root/.cache/bx_ext; do not also wrap in flock(1)")
    ap.add_argument("--ext-cache", default="",
                    help="content-addressed load_inline build root, e.g. /root/.cache/bx_ext")
    ap.add_argument("--pin-pdl8", default=None,
                    help="after prepare set state pdl8_on to these T (e.g. '1,8,16,32' or ''); T = 8 gets every PDL8D edge")
    ap.add_argument("--skip-pdl8-gate", action="store_true",
                    help="with --pin-pdl8: replace the file's in-prepare PDL8 A/B and stress gate (_pdl8_setup) by a no-op")
    ap.add_argument("--profile", action="store_true",
                    help="no timing: one eager call per case inside cudaProfilerStart/Stop (ncu --profile-from-start off)")
    a = ap.parse_args()
    if a.skip_pdl8_gate and a.pin_pdl8 is None:
        ap.error("--skip-pdl8-gate needs --pin-pdl8")
    global LOCK
    if a.lock:
        if outer_flock_holds(a.lock):
            ap.error(f"--lock {a.lock}: a parent flock(1) already holds it; drop the outer flock or --lock")
        a.ext_cache = a.ext_cache or "/root/.cache/bx_ext"
        release_lock_for_triton()
        LOCK = GpuLock(a.lock)
        LOCK.acquire("start")
    if a.ext_cache:
        content_addressed_extensions(a.ext_cache)
    phase("import_torch")

    pdir = os.path.abspath(a.problem)
    name = os.path.basename(pdir.rstrip("/"))
    sys.path.insert(0, pdir)
    ref = load_module("reference", os.path.join(pdir, "reference.py"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.init()
    phase("load_reference_cuda_init")

    router_used = None
    judge, judge_cases = read_judge(a.judge)
    cases = judge_cases or [dict(c, seed=1000 + i) for i, c in enumerate(CASES[name])]
    if a.loads:
        keep = {s.strip() for s in a.loads.split(",")}
        cases = [c for c in cases if c.get("load") in keep]

    weights = ref.generate_weights(ref.WEIGHTS_SEED) if hasattr(ref, "generate_weights") else None
    targets = {}
    for recs in judge.values():
        d = implied_distinct(ref, recs[0])
        if d is not None:
            targets.setdefault(recs[0]["case"]["load"], []).append(d)
    targets = {k: sum(v) / len(v) for k, v in targets.items()}
    if a.router_seed is not None:
        weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=a.router_seed)
        router_used = a.router_seed
    elif targets and hasattr(ref, "_mean_busiest_load"):
        t0 = time.monotonic()
        err, rs, got = distinct_router_seed(ref, weights, cases, targets, ref.WEIGHTS_SEED, a.router_attempts)
        print(f"BOX router_distinct targets {targets}; router_seed {rs} gives {got} (max err {err:.3f},"
              f" {time.monotonic() - t0:.0f} s)", flush=True)
        weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=rs)
        router_used = rs
    torch.cuda.synchronize()
    phase("weights_and_router")

    sides = [("submission", a.submission)]
    if a.baseline:
        sides.append(("baseline", os.path.join(pdir, "baseline.py")))
    impl = {}
    for side, path in sides:
        t0 = time.monotonic()
        mod = load_module(f"okc_{side}", path)
        if a.skip_pdl8_gate and hasattr(mod, "_pdl8_setup"):
            mod._pdl8_setup = lambda st: print("BX in-prepare PDL8 gate skipped (--skip-pdl8-gate)", flush=True)
        state = mod.prepare(weights) if hasattr(mod, "prepare") else None
        gate = pdl8_state(state)
        if a.pin_pdl8 is not None and gate is not None:
            pins = {int(t) for t in a.pin_pdl8.split(",") if t.strip()}
            state["pdl8_on"] = {T: T in pins for T in getattr(mod, "PDL8_T", (1, 8, 16, 32))}
            state["pdl8_edges"] = {8: frozenset(mod.PDL8D_EDGES)} if 8 in pins else {}
        if gate is not None:
            PDL8[side] = dict(prepare=gate, timed=pdl8_state(state))
            print(f"BX {side} pdl8 prepare {gate['on']} timed {PDL8[side]['timed']['on']}", flush=True)
        torch.cuda.synchronize()
        impl[side] = (mod.kernel_fn, state)
        phase(f"prepare_{side}")
        print(f"BOX {side} import+prepare {time.monotonic() - t0:.1f} s", flush=True)

    lsc = (LocalStartCache(a.ls_cache, os.path.join(pdir, "reference.py"), router_used, getattr(ref, "WEIGHTS_SEED", None))
           if a.ls_cache and hasattr(ref, "case_fields") and weights is not None else None)
    timer = Timer()
    rows = []
    for case in cases:
        case = dict(case)
        if hasattr(ref, "case_fields") and weights is not None:
            case.update(ref.case_fields(case, weights))
        seeds = [case["seed"] + j for j in range(a.inputs or a.runs)]
        fresh = []
        for s in seeds:
            # case_fields runs per input on the judge: each fresh input's local_start is its own
            # busiest range, the range the router_load / router_distinct targets are measured on.
            # One case-seed local_start for every input left about a third of decode 1 inputs with
            # no local expert (docs/box.md, "Routing").
            extra = (lsc.get(ref, case, s, weights) if lsc is not None and not a.case_local_start else
                     ref.case_fields({**case, "seed": s}, weights)
                     if hasattr(ref, "case_fields") and weights is not None and not a.case_local_start else {})
            fresh.append(ref.generate_input(**{**case, "seed": s, **extra}))
        row = dict(case=case)
        if lsc is not None:
            lsc.save()
        torch.cuda.synchronize()
        phase("fresh_inputs")
        if a.profile:
            fn, state = impl["submission"]
            data = fresh[0]
            call(fn, data, state)
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            call(fn, data, state)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            print(f"BOX profiled {case.get('load')} {case.get('layer', '')}", flush=True)
            continue
        if not a.no_check:
            data = ref.generate_input(**{**case, "seed": case["seed"] + 7919})
            fn, state = impl["submission"]
            ok, msg = check(ref, data, call(fn, data, state), weights)
            torch.cuda.synchronize()
            row["check"] = [bool(ok), msg]
            del data
            phase("check")
        graphs = {}
        for side, (fn, state) in impl.items():
            static = ref.generate_input(**{**case, "seed": seeds[0]})
            graphs[side] = (capture(fn, static, state), static)
        phase("capture")
        ov = timer.overhead()
        phase("overhead_graphs")
        runs = {side: [] for side in graphs}
        for i in range(a.runs):
            order = list(graphs) if i % 2 == 0 else list(graphs)[::-1]
            for side in order:
                (g, _), static = graphs[side]
                copy_into(static, fresh[i % len(fresh)])
                torch.cuda.synchronize()
                runs[side].append(timer.cold(g))
        phase("replays")
        sub = runs["submission"]
        row.update(replay_overhead=ov, runs_ns=sub, median_raw_ns=statistics.median(sub),
                   median_net_ns=statistics.median(sub) - ov["overhead_ns"])
        if "baseline" in runs:
            base = runs["baseline"]
            net = lambda t: max(t - ov["overhead_ns"], 1.0)  # noqa: E731
            ratios = [net(b) / net(s) for b, s in zip(base, sub)]
            row.update(baseline_runs_ns=base, baseline_raw_ns=statistics.median(base),
                       paired_ratio=statistics.median(ratios),
                       median_ns=(statistics.median(base) - ov["overhead_ns"]) / statistics.median(ratios))
        jr = judge.get(cell_key({k: v for k, v in case.items() if k != "local_start"}), [])
        if jr:
            row["judge_net_us"] = [(r["median_raw_ns"] - r["replay_overhead"]["overhead_ns"]) / 1e3 for r in jr]
            row["judge_shown_us"] = [r["median_ns"] / 1e3 for r in jr]
            row["judge_baseline_net_us"] = [r["baseline_ns"] / 1e3 for r in jr]
        if a.kernels:
            (g, _), static = graphs["submission"]
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for i in range(10):
                    copy_into(static, fresh[i % len(fresh)])
                    torch.cuda.synchronize()
                    timer.cold(g)
            agg = {}
            for ev in prof.events():
                if ev.device_type == torch.autograd.DeviceType.CUDA and "reduce_kernel" not in ev.name and not ev.name.startswith(("Memcpy", "Memset")):
                    agg[ev.name] = agg.get(ev.name, 0.0) + ev.device_time / 10
            row["kernels_us"] = agg
            for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
                print(f"  KERNEL {v:8.2f} us  {k[:90]}", flush=True)
            print(f"  KERNEL {sum(agg.values()):8.2f} us  total", flush=True)
            phase("kernels_profiler")
        rows.append(row)
        label = f"{case.get('load')} {case.get('layer', '')}".strip()
        line = (f"BOX {label}: raw {row['median_raw_ns'] / 1e3:.2f} net {row['median_net_ns'] / 1e3:.2f} us"
                f" (overhead {ov['overhead_ns'] / 1e3:.2f})")
        if "paired_ratio" in row:
            line += (f" | sglang raw {row['baseline_raw_ns'] / 1e3:.2f} paired {row['paired_ratio']:.3f}"
                     f" shown {row['median_ns'] / 1e3:.2f}")
        if jr:
            jn = statistics.median(row["judge_net_us"])
            line += (f" | judge net {jn:.2f} (n={len(jr)}, {min(row['judge_net_us']):.2f}-"
                     f"{max(row['judge_net_us']):.2f}) box/judge {row['median_net_ns'] / 1e3 / jn:.3f}")
        if jr and "baseline_raw_ns" in row:
            jb = statistics.median(row["judge_baseline_net_us"])
            bb = (row["baseline_raw_ns"] - ov["overhead_ns"]) / 1e3
            line += f" | sglang box/judge {bb:.2f}/{jb:.2f} = {bb / jb:.3f}"
        if "check" in row:
            line += f" | check {'PASS' if row['check'][0] else 'FAIL'}"
        print(line, flush=True)
        del graphs, fresh
        torch.cuda.empty_cache()
        phase("cell_tail")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(problem=name, submission=a.submission, gpu=torch.cuda.get_device_name(),
                           torch=torch.__version__, rows=rows, phases=PH, pdl8=PDL8, argv=sys.argv,
                           lock_events=LOCK.events if LOCK else None), f)
    phase("json_write")
    if lsc is not None:
        print(f"BX local_start cache {a.ls_cache}: {lsc.hits} hits, {lsc.misses} computed", flush=True)
    print("BXPHASE " + " ".join(f"{k}={v:.2f}" for k, v in PH.items()) + f" total={time.monotonic() - _T0:.2f}", flush=True)


if __name__ == "__main__":
    main()
