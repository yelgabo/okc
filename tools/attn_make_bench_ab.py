"""Write a paired A/B bench for the one-launch kernel: submission.py (the judged path is the
control) plus extra builds of attn_one.cu with compile flags, and optionally the public entry's
kernel loaded from its own file, all timed in one process on the same inputs.

    python3 tools/attn_make_bench_ab.py OUT.py [--pub [NAME=]PUBLIC.py ...] [--null]
        [--shapes sliding|full|all] [--seed N] [--inputs K] [--rounds R] [--nocoop]
        [--set NAME=VALUE ...] NAME[@own]=[FLAGS][:mode=M][:wsoff=BYTES][:phase=K][:early][:pick=K] ...

NAME=FLAGS adds a variant built from submission.py's inlined _ONE_SRC (not attn_one.cu) with
those nvcc flags (comma-separated), e.g. diet=-DONE_DIAG=0,-DONE_DIET=1. A switch that exists
only in attn_one.cu needs `tools/attn_inline_one.py OUT.py` on the written bench. The variant
runs through _one with its module swapped in. By default it shares the control's workspace
and residency table (the plan comes from the control's prepare); NAME@own=FLAGS gives the arm
its own state: its own ws_acc / ws_l / ws_cnt, its own attn_one_prepare residency query and
cooperative-launch probe, so its plan and workspace history are independent of the control's.
A `:mode=M` suffix forces launch mode M (e.g. s115=:mode=115, the control's flags at one CTA per
SM; s115x=-DONE_XPIPE=1:mode=115) on the shapes where the control's plan is a flat q_len 8
mode (51 or 19), through _one(data, mode=M), which sizes the grid from that instance's
residency; elsewhere the arm runs the normal plan. ABPLAN marks the forced plans. A forced-mode
arm always gets its own state (as @own).

Arms with the same flags share one build; an arm with no flags reuses the control's module (no
compile), except ctrl2. `:wsoff=BYTES` gives the arm its own state whose ws_acc, ws_l and ws_cnt
start BYTES into larger zeroed allocations (a placement probe: same code, other addresses).
`:pick=K` (own state) sets submission's _ONE_WS_PICK = K while the arm runs, so its state is
created by timing K candidate workspaces on the batch-1 full plans and keeping the fastest;
ABADDR then shows the pick and the candidates' medians.
`:tkt` (build with -DONE_T9=1) runs the last-CTA mode in place of the control's cooperative
per-request mode (3 -> 1, 11 -> 9, 15 -> 13) with the same splits and copies; ABPLAN shows
tkt_mode. `:phase=K` (needs a -DONE_DIAG=1 build, own state, name ending in _probe) runs the diagnostic
early exit after phase K of the per-request path (attn_one.cu PHASE_EXIT) on the shapes whose
control plan is per-request (no flat bit), and the full call elsewhere; its cold time is the
time to that phase. `:early` (own state) creates the arm's state at import, before the
control's first call allocates the control's. `--set NAME=VALUE` assigns a submission global
before any call (e.g. --set _ONE_WS_CARVE=4096). ABADDR, printed with the summaries, gives each
arm's workspace addresses.

--null adds the independent duplicate controls: `ctrl2`, a second build of the control's
source and flags under its own extension name with its own state (as @own), and for every
`--pub` arm not named pubstamp* or *_probe, a second copy `<name>2` executed into its own module object
with its load_inline names rewritten, so it compiles a separate extension and keeps its own
workspace. ctrl2 - ctrl and pub2 - pub are the nulls: what two independently prepared copies
of the same code measure apart in this process.

At import the bench prints, before the judge's own tests and timing:

  ABCHECK   per variant and shape (scored shapes and seven tp8 test shapes, input seed 1): for
            the control the judge-style check against the FP32 reference, for every other
            variant its output against the control's (relative L2 below 2e-2, differing
            elements), and the workspace invariant (each arm's own workspace) after the call.
  ABPLAN    per timed shape: every arm's selected plan, side by side.
  ABCOLD    per scored shape and timing input: median of R graph replays (default 120) after a
            256 MB L2 flush; each round runs the arms in a permutation drawn from
            random.Random(seed + round). Timing inputs are held out from ABCHECK: seeds
            1000 + k for k < K (--inputs, default 1). With K > 1 a `pooled` row follows.
  ABWARM    per scored shape and input: a graph of 16 back-to-back calls, median over 25
            replays, divided by 16 (L2 warm, launch gaps of a captured graph), same order rule.
  ABSUM     at the end, one cold and one warm line per shape (pooled over inputs): each arm's
            median, the nulls (ctrl2-ctrl, pub2-pub) and every other arm minus ctrl.
  ABIN      at the end, per scored shape and input: every non-probe arm's cold median minus
            ctrl's.
  ABSCORE   at the end, when all twelve shapes ran: per arm the standalone score ratio against
            ctrl from the cold medians, prod_l (C_ctrl / C_arm) ^ (1/6) with C = 46 t_sliding
            + 16 t_full per load, and the same per load.

Arms whose name ends in `_probe` are timing probes with wrong or unguaranteed output: ABCHECK
prints PROBE for them and does not count them as failures. An arm whose name starts with `stamp`
(built with -DONE_STAMP=1) also gets STAMP and STAMPWAVE lines: per-CTA records of five cold
calls per flat full-context shape, q_len 1 and 8 (attn_one.cu, ONE_STAMP). `--nocoop` adds the arm `nocoop`:
the control's module launched without cudaLaunchAttributeCooperative, same grid.

A public arm whose name starts with `pubstamp` (a copy of the public file with globaltimer
stamps written to dbg[8192 + 8 c ..]) gets PUBSTAMP and STAMPWAVE lines: the same loop-end
spread by dispatch wave and by SM.

--pubsub NAME=BASE:OLD=>NEW adds a public arm whose source is BASE's with the unique text OLD
replaced by NEW at import (a backslash-n in OLD / NEW is a newline), so variants of one public
file are embedded once.

--pub may repeat; each public file is an arm (default name `pub`; its names must start with
`pub`, and a second file needs its own load_inline name). --shapes full keeps the full-context
shapes with batch > 1 (modes 49 and 51); all (the default) keeps the twelve scored shapes. The
public files are read at generation time and embedded as a string; the output belongs in the
session scratchpad, not in the repository (only the logs of its runs are committed). The judge
keeps the last 16,000 characters of the log, so the summaries are repeated at the end and
ABCHECKSUM is printed last.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import attn_make_bench_one as base  # noqa: E402

# Plain template: the configuration replaces the @@CONFIG@@ line, so braces are literal.
AB = r'''

import contextlib as _ab_contextlib
import random as _ab_random

@@CONFIG@@
_AB_MODS = {}
_AB_BUILT = {}
for _name, _fl in _AB_FLAGS.items():
    _key = tuple(_fl)
    if _name != "ctrl2" and not _fl:
        _AB_MODS[_name] = _ONE
        print(f"ABBUILD {_name} reuses the control module own_state={_name in _AB_OWN}",
              flush=True)
        continue
    if _name != "ctrl2" and _key in _AB_BUILT:
        _AB_MODS[_name] = _AB_BUILT[_key]
        print(f"ABBUILD {_name} {_fl} shares a build own_state={_name in _AB_OWN}", flush=True)
        continue
    try:
        _AB_MODS[_name] = load_inline(
            name=f"okc_attn_one_ab_{_name}",
            cpp_sources=_ONE_CPP,
            cuda_sources=_ONE_SRC,
            functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", *_fl],
        )
        if _name != "ctrl2":
            _AB_BUILT[_key] = _AB_MODS[_name]
        print(f"ABBUILD {_name} {_fl} own_state={_name in _AB_OWN} loaded {_ob_elapsed()}",
              flush=True)
    except Exception as _exc:
        print(f"ABBUILD {_name} FAIL {type(_exc).__name__}: {str(_exc)[:400]}", flush=True)
for _name, (_base, _old, _new) in _AB_PUB_SUBS.items():
    if _AB_PUB_SRCS[_base].count(_old) != 1:
        raise SystemExit(f"--pubsub {_name}: {_old!r} is not unique in {_base}")
    _AB_PUB_SRCS[_name] = _AB_PUB_SRCS[_base].replace(_old, _new)
_AB_PUBS = {}
for _name, _src in _AB_PUB_SRCS.items():
    try:
        import types as _types
        _m = _types.ModuleType(f"okc_ab_{_name}")
        exec(compile(_src, f"{_name}_attention.py", "exec"), _m.__dict__)
        _m._module()
        _AB_PUBS[_name] = _m
        print(f"ABBUILD {_name} loaded {_ob_elapsed()}", flush=True)
    except Exception as _exc:
        print(f"ABBUILD {_name} FAIL {type(_exc).__name__}: {str(_exc)[:400]}", flush=True)
# Own-state arms: a private _ONE_STATE dict per arm, swapped in around each call.
_AB_STATES = {name: {} for name in _AB_OWN}
_AB_WSDONE = set()
for _name, _val in _AB_SET.items():
    globals()[_name] = _val


@_ab_contextlib.contextmanager
def _ab_ctx(name):
    """Run as arm `name`: its module in _ONE and, for own-state arms, its state dict."""
    global _ONE, _ONE_STATE, _ONE_WS_PICK
    mod = _AB_MODS.get(name)
    if mod is None:
        yield
        return
    saved = (_ONE, _ONE_STATE, _ONE_WS_PICK)
    _ONE = mod
    _ONE_WS_PICK = _AB_PICK.get(name, _ONE_WS_PICK)
    if name in _AB_STATES:
        _ONE_STATE = _AB_STATES[name]
    try:
        off = _AB_WSOFF.get(name)
        if off is not None and name not in _AB_WSDONE:
            dev = torch.device("cuda", torch.cuda.current_device())
            st = _one_state(dev)
            for key in ("acc", "l", "cnt"):
                ref = st[key]
                pad = off // ref.element_size()
                big = torch.zeros(ref.numel() + pad, dtype=ref.dtype, device=dev)
                st[key] = big[pad:pad + ref.numel()].view(ref.shape)
                st["ab_keep_" + key] = big
            _AB_WSDONE.add(name)
            print(f"ABWSOFF {name} off={off} acc=0x{st['acc'].data_ptr():x} "
                  f"l=0x{st['l'].data_ptr():x} cnt=0x{st['cnt'].data_ptr():x}", flush=True)
        yield
    finally:
        _ONE, _ONE_STATE, _ONE_WS_PICK = saved


for _name in _AB_EARLY:
    with _ab_ctx(_name):
        _one_state(torch.device("cuda", torch.cuda.current_device()))
    print(f"ABEARLY {_name} state created at import", flush=True)


def _ab_addr():
    dev = torch.device("cuda", torch.cuda.current_device())
    parts = []
    for n in ["ctrl"] + [x for x in _AB_MODS if x in _AB_STATES or x == "ctrl2"]:
        st = (_ONE_STATE if n == "ctrl" else _AB_STATES.get(n, {})).get(dev)
        if st:
            pick = st.get("ws_pick")
            parts.append(f"{n}:acc=0x{st['acc'].data_ptr():x},l=0x{st['l'].data_ptr():x},"
                         f"cnt=0x{st['cnt'].data_ptr():x}"
                         + (f",pick={pick[0]}:" + "/".join(f"{m:.2f}" for m in pick[1])
                            if pick else ""))
    print("ABADDR " + " ".join(parts), flush=True)


def _ab_forced(name, data):
    """The mode arm `name` forces on this shape: only where the control's own plan is a flat
    q_len 8 mode (51 or 19); None elsewhere, where the arm runs the normal plan."""
    m = _AB_MODES.get(name)
    if m is None:
        return None
    cfg = data["config"]
    ctrl_mode = _one_plan(cfg, data["page_table"].shape[1],
                          _one_state(data["qkv"].device)["resident"])[2]
    return m if ctrl_mode in (51, 19) else None


def _ab_phase(name, data):
    k = _AB_PHASE.get(name)
    if not k:
        return 0
    cfg = data["config"]
    ctrl_mode = _one_plan(cfg, data["page_table"].shape[1],
                          _one_state(data["qkv"].device)["resident"])[2]
    return 0 if ctrl_mode & 16 else k


def _ab_tkt(name, data):
    """The last-CTA mode an arm with :tkt runs where the control's plan is a cooperative
    per-request mode (3 -> 1, 11 -> 9, 15 -> 13; the build needs -DONE_T9=1); None elsewhere."""
    if name not in _AB_TKT:
        return None
    cfg = data["config"]
    ctrl_mode = _one_plan(cfg, data["page_table"].shape[1],
                          _one_state(data["qkv"].device)["resident"])[2]
    return ctrl_mode ^ 2 if (ctrl_mode & 3) == 3 and not ctrl_mode & 16 else None


def _ab_via(name):
    def call(data):
        m = _ab_forced(name, data) or _ab_tkt(name, data)
        k = _ab_phase(name, data)
        with _ab_ctx(name):
            if k:
                return _one(data, phase=k)
            return _one(data, mode=m) if m else _one(data)
    return call


def _ab_nocoop(data):
    st = _one_state(data["qkv"].device)
    saved = st["coop_launch"]
    st["coop_launch"] = False
    try:
        return _one(data)
    finally:
        st["coop_launch"] = saved


def _ab_fns():
    fns = {"ctrl": _one}
    if _AB_NOCOOP:
        fns["nocoop"] = _ab_nocoop
    for name in _AB_MODS:
        fns[name] = _ab_via(name)
    for name, mod in _AB_PUBS.items():
        fns[name] = mod.kernel_fn
    return fns


def _ab_scored():
    for batch, layer, q_len in _OB_SCORED:
        if _AB_SHAPES == "sliding" and layer != "sliding":
            continue
        if _AB_SHAPES == "full" and (layer == "sliding" or batch == 1):
            continue
        if _AB_SHAPES == "b1" and batch != 1:
            continue
        yield batch, layer, q_len


def _ab_shape_inputs(batch, layer, q_len, seed):
    return _ob_inputs(batch, 80000, "tp8", layer, q_len, "fixed" if batch == 1 else "agentic",
                      "soft" if layer == "sliding" else "coherent", 1, 1.0, seed=seed)


def _ab_plan(name, data):
    try:
        if name.startswith("pub"):
            mod = _AB_PUBS[name]
            cfg = data["config"]
            pairs = cfg["num_kv_heads"] // 2
            plan = mod._plan(cfg["batch"], pairs, cfg["q_len"], cfg["window"],
                             data["page_table"].shape[1], cfg["page_size"], data["qkv"].device)
            tuned = getattr(mod, "_TUNED", {}).get(
                (cfg["batch"], cfg["q_len"], bool(cfg["window"]), pairs))
            if tuned:
                plan = dict(plan, **tuned)
            return ",".join(f"{k}={v}" for k, v in plan.items())
        if name == "nocoop":
            return _ob_plan_str(data).replace(" ", ",") + ",nocoop"
        m = _ab_forced(name, data)
        t = _ab_tkt(name, data)
        with _ab_ctx(name):
            if t:
                return _ob_plan_str(data).replace(" ", ",") + f",tkt_mode={t}"
            if m:
                res = _one_state(data["qkv"].device)["resident"]
                n = min(_ONE_TARGET, res.get((data["config"]["q_len"], m), 0))
                return f"splits={n},nc=1,mode={m},forced"
            return _ob_plan_str(data).replace(" ", ",")
    except Exception as exc:
        return f"plan?{type(exc).__name__}"


_AB_TALLY = {}


def _ab_check():
    fns = _ab_fns()
    dev = torch.device("cuda", torch.cuda.current_device())
    shapes = [(b, 80000, "tp8", lay, q, "fixed" if b == 1 else "agentic",
               "soft" if lay == "sliding" else "coherent", 1, 1.0) for b, lay, q in _ab_scored()]
    # The tp8 test shapes that reach the paths a variant can change (page size 64, kv scale,
    # mixed lengths, short contexts), without the 1M and 80K test contexts, whose FP32
    # references would cost minutes of the run's 15.
    shapes += [_OB_TESTS[i] for i in (0, 1, 2, 3, 7, 9, 10)]
    fails = 0
    for shape in shapes:
        name = "b{} ctx{} {} {} L{} {} {} page{} scale{:g}".format(*shape)
        try:
            data = _ob_inputs(*shape)
            if not _one_ok(data["config"]):
                continue
            ref = None
            for vname, fn in fns.items():
                out = _ob_run(lambda: fn(data), f"{vname} {name}")
                torch.cuda.synchronize()
                if vname.startswith("pub"):
                    wok, wmsg = True, ""
                else:
                    with _ab_ctx(vname):
                        wok, wmsg = _ob_ws_check(dev)
                plan = _ab_plan(vname, data)
                # The judge-style check (two FP32 references) for the control only; the
                # variants are compared with the control's output.
                if ref is None:
                    ok, msg = _ob_check(data, out)
                    ref = out.float().clone()
                    vs = ""
                else:
                    rel = _ob_rel(out, ref)
                    ok, msg = rel < 2e-2, ""
                    vs = f"vs_ctrl={rel:.1e} {_ob_diffstat(out, ref)}"
                status = "PASS" if ok and wok else "FAIL"
                if vname.endswith("_probe"):
                    status = "PROBE"
                fails += status == "FAIL"
                tally = _AB_TALLY.setdefault(vname, {"PASS": 0, "FAIL": 0, "PROBE": 0})
                tally[status] += 1
                print(f"ABCHECK {status} {vname} {name} [{plan}] {msg}{vs} {wmsg}", flush=True)
            data = None
        except Exception as exc:
            fails += 1
            print(f"ABCHECK FAIL {name} {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        torch.cuda.empty_cache()
    print(f"ABCHECK done: {fails} failing {_ob_elapsed()}", flush=True)


def _ab_graph(fn, data, calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(data)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn(data)
    torch.cuda.synchronize()
    return graph


def _ab_med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def _ab_order(names, r):
    order = list(names)
    _ab_random.Random(_AB_SEED + r).shuffle(order)
    return order


# (tag, kind) -> {arm: [samples pooled over inputs]}; kind is "cold" or "warm".
_AB_POOL = {}
# itag -> the arm whose cold graph was captured first.
_AB_CAP = {}
# (tag, input) -> {arm: cold median}, for the compact ABIN lines at the end.
_AB_INMED = {}


def _ab_bench(warm_calls=16, warm_reps=25):
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    fns = _ab_fns()
    names = list(fns)
    print(f"ABORDER seed={_AB_SEED} rounds={_AB_ROUNDS} inputs={_AB_INPUTS} "
          f"timing_seeds={[1000 + k for k in range(_AB_INPUTS)]} arms={names}", flush=True)
    for batch, layer, q_len in _ab_scored():
        tag = f"b={batch:2d} {layer:7s} L={q_len}"
        cold_pool = _AB_POOL.setdefault((tag, "cold"), {n: [] for n in names})
        warm_pool = _AB_POOL.setdefault((tag, "warm"), {n: [] for n in names})
        for k in range(_AB_INPUTS):
            data = _ab_shape_inputs(batch, layer, q_len, 1000 + k)
            itag = f"{tag} in={k}"
            try:
                # Capture order seeded per (shape, input): checks a0 / b / c had the control
                # captured first and slower at d1 full by 0.45-0.74 us than every other arm.
                cap = _ab_order(names, 7907 + 101 * len(_AB_CAP))
                _AB_CAP[itag] = cap[0]
                cold = {n: _ab_graph(fns[n], data, 1) for n in cap}
                if k == 0:
                    print(f"ABPLAN {tag} " + " ".join(f"{n}=[{_ab_plan(n, data)}]" for n in names),
                          flush=True)
                samples = {n: [] for n in names}
                for r in range(_AB_ROUNDS):
                    evs = []
                    for n in _ab_order(names, r):
                        flush.zero_()
                        a = torch.cuda.Event(enable_timing=True)
                        z = torch.cuda.Event(enable_timing=True)
                        a.record()
                        cold[n].replay()
                        z.record()
                        evs.append((n, a, z))
                    torch.cuda.synchronize()
                    for n, a, z in evs:
                        samples[n].append(1000.0 * a.elapsed_time(z))
                cold = None
                for n in names:
                    cold_pool[n] += samples[n]
                _AB_INMED[(tag, k)] = {n: _ab_med(samples[n]) for n in names}
                row = " ".join(f"{n}={_ab_med(samples[n]):.2f}" for n in names)
                print(f"ABCOLD {itag} {row} cap0={_AB_CAP[itag]} {_ob_elapsed()}", flush=True)
                warm = {n: _ab_graph(fns[n], data, warm_calls) for n in names}
                ws = {n: [] for n in names}
                for r in range(warm_reps):
                    evs = []
                    for n in _ab_order(names, 7919 + r):
                        a = torch.cuda.Event(enable_timing=True)
                        z = torch.cuda.Event(enable_timing=True)
                        a.record()
                        warm[n].replay()
                        z.record()
                        evs.append((n, a, z))
                    torch.cuda.synchronize()
                    for n, a, z in evs:
                        ws[n].append(1000.0 * a.elapsed_time(z) / warm_calls)
                warm = None
                for n in names:
                    warm_pool[n] += ws[n]
                row = " ".join(f"{n}={_ab_med(ws[n]):.2f}" for n in names)
                print(f"ABWARM {itag} {row} {_ob_elapsed()}", flush=True)
            except Exception as exc:
                print(f"ABBENCH FAIL {itag} {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            data = None
            torch.cuda.empty_cache()
        if _AB_INPUTS > 1:
            for kind, pool in (("ABCOLD", cold_pool), ("ABWARM", warm_pool)):
                row = " ".join(f"{n}={_ab_med(pool[n]):.2f}" for n in names)
                print(f"{kind} {tag} pooled {row}", flush=True)


def _ab_summary():
    names = list(_ab_fns())
    nulls = [("ctrl2", "ctrl")] + [(n, n[:-1]) for n in names
                                   if n.startswith("pub") and n.endswith("2") and n[:-1] in names]
    null_arms = {a for a, _ in nulls}
    meds = {}
    for (tag, kind), pool in _AB_POOL.items():
        m = {n: _ab_med(pool.get(n, [])) for n in names}
        meds[(tag, kind)] = m
        if kind == "warm" and len(names) > 8:
            # Keeps the cold summary inside the judge's 16,000-character log tail; the
            # per-input ABWARM lines carry the warm medians.
            continue
        parts = [f"{n}={m[n]:.2f}" for n in names]
        parts += [f"null:{a}-{b}={m[a] - m[b]:+.2f}" for a, b in nulls if a in m and b in m]
        parts += [f"d:{n}={m[n] - m['ctrl']:+.2f}" for n in names
                  if n != "ctrl" and n not in null_arms]
        print(f"ABSUM {kind} {tag} " + " ".join(parts), flush=True)
    # Per-input cold differences against ctrl, short enough to survive the log tail.
    for (tag, k), m in _AB_INMED.items():
        print(f"ABIN {tag} in={k} " + " ".join(
            f"{n}={m[n] - m['ctrl']:+.2f}" for n in names
            if n != "ctrl" and not n.endswith("_probe")), flush=True)
    # Standalone score ratio against ctrl from the cold medians (TASK.md: C = 46 t_sliding +
    # 16 t_full per load, geometric mean over the six loads).
    loads = [(b, q) for b, lay, q in _OB_SCORED if lay == "sliding"]
    cold = {}
    for b, q in loads:
        ts = meds.get((f"b={b:2d} sliding L={q}", "cold"))
        tf = meds.get((f"b={b:2d} full    L={q}", "cold"))
        if ts is None or tf is None:
            return
        cold[(b, q)] = {n: 46.0 * ts[n] + 16.0 * tf[n] for n in names}
    for n in names:
        prod = 1.0
        per = []
        for b, q in loads:
            r = cold[(b, q)]["ctrl"] / cold[(b, q)][n]
            prod *= r
            per.append(f"{'d' if q == 1 else 'v'}{b}={r:.4f}")
        print(f"ABSCORE {n} vs ctrl {prod ** (1.0 / len(loads)):.4f} " + " ".join(per),
              flush=True)


_ST_BASE, _ST_W = 1024, 16
# Per (shape, arm): statistic -> one value per repetition; STSUM prints their medians at the end
# of the log, where the judge's 16,000-character tail keeps them.
_ST_SUM = {}


def _st_add(tag, name, **vals):
    d = _ST_SUM.setdefault((tag, name), {})
    for k, v in vals.items():
        d.setdefault(k, []).append(float(v))


def _st_q(xs):
    xs = sorted(float(x) for x in xs)
    return xs[0], xs[len(xs) // 2], xs[-1]


def _ab_stsum():
    for (tag, name), d in _ST_SUM.items():
        med = {k: sorted(v)[len(v) // 2] for k, v in d.items()}
        print(f"STSUM {tag} {name} reps={len(next(iter(d.values())))} " + " ".join(
            f"{k}={v:.1f}" for k, v in med.items()), flush=True)


def _ab_wave(tag, name, rep, ends, sms):
    """Loop-end spread by dispatch wave and by SM: ends[c] in us for CTA c, sms[c] its SM."""
    n = len(ends)
    h = n // 2

    def q(xs):
        xs = sorted(xs)
        if not xs:
            return "-"
        return "{:.1f}/{:.1f}/{:.1f}".format(xs[0], xs[len(xs) // 2], xs[-1])
    by_sm = {}
    for c in range(n):
        by_sm.setdefault(int(sms[c]), []).append(c)
    pairs = [sorted(v) for v in by_sm.values() if len(v) == 2]
    gaps = [ends[b] - ends[a] for a, b in pairs]
    low_first = sum(1 for g in gaps if g > 0)
    sm_end = [max(ends[c] for c in v) for v in by_sm.values()]
    if sm_end:
        _st_add(tag, name, smend_med=_st_q(sm_end)[1], smend_max=_st_q(sm_end)[2])
    if gaps:
        _st_add(tag, name, pairgap_med=_st_q([abs(g) for g in gaps])[1])
    print(f"STAMPWAVE {tag} {name} rep={rep} ctas={n} sms={len(by_sm)} pairs={len(pairs)} "
          f"wave1_end={q(ends[:h])} wave2_end={q(ends[h:])} "
          f"pair_gap_hi_minus_lo={q(gaps)} abs_gap={q([abs(g) for g in gaps])} "
          f"lo_first={low_first}/{len(pairs)} sm_end={q(sm_end)} "
          f"mean_end={sum(ends) / max(n, 1):.1f} max_end={max(ends):.1f}", flush=True)


def _ab_pubstamps(reps=5):
    """Per-CTA records of the public flat kernel's stamp copy (arms named pubstamp*): dbg
    int64 [8192 + 8 c + k]: 0 start, 1 consumer loop end, 2 finalize done (globaltimer ns),
    3 SM id, 4 tiles, 5 segments."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for name, mod in _AB_PUBS.items():
        if not name.startswith("pubstamp"):
            continue
        for batch, layer, q_len in _ab_scored():
            if layer == "sliding" or batch == 1:
                continue
            data = _ob_inputs(batch, 80000, "tp8", layer, q_len, "agentic", "coherent", 1, 1.0)
            tag = f"b={batch:2d} {layer:7s} L={q_len}"
            for r in range(reps):
                buf = torch.zeros(8192 + 8 * 1024, dtype=torch.int64, device="cuda")
                flush.zero_()
                mod._launch(data, dbg=buf)
                torch.cuda.synchronize()
                rec = buf[8192:].view(1024, 8).cpu()
                n = int((rec[:, 0] != 0).sum())
                rec = rec[:n]
                ref = int(rec[:, 0].min())
                start = [(int(x) - ref) / 1000.0 for x in rec[:, 0]]
                # q_len 1 copies (field 6 set): loop end 6, kernel end 1.
                lcol, fcol = (6, 1) if int(rec[:, 6].min()) > 0 else (1, 2)
                ends = [(int(x) - ref) / 1000.0 for x in rec[:, lcol]]
                fin = [(int(x) - ref) / 1000.0 for x in rec[:, fcol]]
                tails = [f - e for f, e in zip(fin, ends)]
                _st_add(tag, name, start_max=_st_q(start)[2], loop_min=_st_q(ends)[0],
                        loop_med=_st_q(ends)[1], loop_max=_st_q(ends)[2],
                        tail_med=_st_q(tails)[1], tail_max=_st_q(tails)[2],
                        kend_med=_st_q(fin)[1], span=max(fin))

                def q(xs):
                    xs = sorted(xs)
                    return "{:.1f}/{:.1f}/{:.1f}".format(xs[0], xs[len(xs) // 2], xs[-1])
                print(f"PUBSTAMP {tag} {name} rep={r} ctas={n} span={max(fin):.1f} "
                      f"start={q(start)} loop_end={q(ends)} "
                      f"barrier_fin={q([f - e for f, e in zip(fin, ends)])} "
                      f"tiles={q([int(x) for x in rec[:, 4]])} segs={q([int(x) for x in rec[:, 5]])} "
                      f"{_ob_elapsed()}", flush=True)
                _ab_wave(tag, name, r, ends, [int(x) for x in rec[:, 3]])
                if q_len == 1:
                    _ab_wave(tag, name + "_kend", r, fin, [int(x) for x in rec[:, 3]])
                if r == 0:
                    order = sorted(range(n), key=lambda i: ends[i])
                    pick = order[:4] + order[n // 2 - 2:n // 2 + 2] + order[-4:]
                    print(f"PUBSTAMPCTA {tag} {name} " + " ".join(
                        "c{}:sm{}:t{}:end{:.1f}".format(i, int(rec[i, 3]), int(rec[i, 4]), ends[i])
                        for i in pick), flush=True)
            data = None
            torch.cuda.empty_cache()


def _ab_stamps(reps=5):
    """Per-CTA records of the ONE_STAMP build on the flat full-context shapes, cold calls.
    Fields: 0 start, 1 partition done, 2 first tile ready, 3 tile loops done (q_len 1: the last
    segment's), 4 grid barrier passed (q_len 1: last segment's reds issued, before its ticket),
    5 finalize done (q_len 1: kernel end) (globaltimer ns, low 32 bits); 6 tiles, 7 segments, 8 consumer
    full-barrier wait, 9 segment prologues, 10 tile loops, 11 producer empty-barrier wait,
    12 producer lifetime, 13 / 14 consumer cycles to loop end / to kernel end; 15 SM id."""
    for sname in _AB_MODS:
        if sname.startswith("stamp"):
            _ab_stamps_one(sname, reps)


def _ab_stamps_one(sname, reps):
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for batch, layer, q_len in _ab_scored():
        if layer == "sliding" or batch == 1:
            continue
        data = _ob_inputs(batch, 80000, "tp8", layer, q_len, "agentic", "coherent", 1, 1.0)
        tag = f"b={batch:2d} {layer:7s} L={q_len}"
        for r in range(reps):
            buf = torch.zeros(_ST_BASE + 512 * _ST_W, dtype=torch.int32, device="cuda")
            flush.zero_()
            with _ab_ctx(sname):
                _one(data, dbg=buf)
                n = _one_plan(data["config"], data["page_table"].shape[1],
                              _one_state(data["qkv"].device)["resident"])[0]
            torch.cuda.synchronize()
            rec = buf[_ST_BASE:_ST_BASE + n * _ST_W].view(n, _ST_W).cpu().long()
            ref = int(rec[:, 0].min())
            ns = ((rec[:, :6] - ref + (1 << 31)) % (1 << 32)) - (1 << 31)
            ns = ns.double() / 1000.0
            cyc = rec[:, 8:15].double()
            ghz = float((cyc[:, 5] / (ns[:, 3] - ns[:, 0]).clamp(min=1e-3) / 1000.0).median())
            us = cyc / (ghz * 1000.0)

            def q(x):
                xs = sorted(float(v) for v in x)
                return "{:.1f}/{:.1f}/{:.1f}".format(xs[0], xs[len(xs) // 2], xs[-1])
            lend = [float(x) for x in ns[:, 3]]
            kend = [float(x) for x in ns[:, 5]]
            _st_add(tag, sname, start_max=_st_q(ns[:, 0])[2],
                    first_med=_st_q(ns[:, 2] - ns[:, 0])[1], loop_min=_st_q(lend)[0],
                    loop_med=_st_q(lend)[1], loop_max=_st_q(lend)[2],
                    mid_med=_st_q(ns[:, 4] - ns[:, 3])[1], mid_max=_st_q(ns[:, 4] - ns[:, 3])[2],
                    tail_med=_st_q([k - e for k, e in zip(kend, lend)])[1],
                    tail_max=_st_q([k - e for k, e in zip(kend, lend)])[2],
                    kend_med=_st_q(kend)[1], span=max(kend),
                    fullwait_med=_st_q(us[:, 0])[1], prodwait_med=_st_q(us[:, 3])[1])
            _ab_wave(tag, sname, r, [float(x) for x in ns[:, 3]], [int(x) for x in rec[:, 15]])
            if q_len == 1:
                _ab_wave(tag, sname + "_kend", r, [float(x) for x in ns[:, 5]],
                         [int(x) for x in rec[:, 15]])
            print(f"STAMP {tag} {sname} rep={r} ctas={n} ghz={ghz:.3f} "
                  f"span={float(ns[:, 5].max()):.1f} start={q(ns[:, 0])} "
                  f"part={q(ns[:, 1] - ns[:, 0])} first_tile={q(ns[:, 2] - ns[:, 0])} "
                  f"loop_end={q(ns[:, 3])} barrier_wait={q(ns[:, 4] - ns[:, 3])} "
                  f"fin={q(ns[:, 5] - ns[:, 4])} tiles={q(rec[:, 6])} segs={q(rec[:, 7])} "
                  f"fullwait_us={q(us[:, 0])} prologue_us={q(us[:, 1])} "
                  f"loop_us={q(us[:, 2])} prod_emptywait_us={q(us[:, 3])} "
                  f"prod_life_us={q(us[:, 4])} {_ob_elapsed()}", flush=True)
            if r == 0:
                # Loop-end time next to tiles, so imbalance can be told from per-tile speed.
                order = sorted(range(n), key=lambda i: float(ns[i, 3]))
                pick = order[:4] + order[n // 2 - 2:n // 2 + 2] + order[-4:]
                print(f"STAMPCTA {tag} {sname} " + " ".join(
                    "c{}:sm{}:t{}:s{}:end{:.1f}:pro{:.1f}:fw{:.1f}".format(
                        i, int(rec[i, 15]), int(rec[i, 6]), int(rec[i, 7]), float(ns[i, 3]),
                        float(us[i, 1]), float(us[i, 0])) for i in pick), flush=True)
        data = None
        torch.cuda.empty_cache()


if not _os.environ.get("ATTN_AB_DONE"):
    _os.environ["ATTN_AB_DONE"] = "1"
    global_ok = _ONE is not None
    if global_ok:
        _ONE_DBG = None
        try:
            _ab_check()
        except Exception as _exc:
            print(f"ABCHECK failed: {type(_exc).__name__}: {_exc}", flush=True)
        try:
            _ab_stamps()
        except Exception as _exc:
            print(f"STAMP failed: {type(_exc).__name__}: {_exc}", flush=True)
        try:
            _ab_pubstamps()
        except Exception as _exc:
            print(f"PUBSTAMP failed: {type(_exc).__name__}: {_exc}", flush=True)
        try:
            _ab_bench()
        except Exception as _exc:
            print(f"ABBENCH failed: {type(_exc).__name__}: {_exc}", flush=True)
        # Repeated last: the run log keeps only its final 16,000 characters.
        try:
            _ab_stsum()
        except Exception as _exc:
            print(f"STSUM failed: {type(_exc).__name__}: {_exc}", flush=True)
        try:
            _ab_addr()
            _ab_summary()
        except Exception as _exc:
            print(f"ABSUM failed: {type(_exc).__name__}: {_exc}", flush=True)
        print("ABCHECKSUM " + " ".join(f"{n}={t['PASS']}/{t['FAIL']}/{t['PROBE']}"
                                       for n, t in _AB_TALLY.items()) + " (pass/fail/probe)",
              flush=True)
'''


def _dup_pub(src, name):
    """A second copy of a public file whose load_inline extension names differ, so exec'ing it
    compiles and loads a separate extension instead of reusing the first copy's."""
    out, n = re.subn(r'(\bname\s*=\s*f?")', r'\g<1>' + name + '_', src)
    if n == 0:
        raise SystemExit(f"--null: no load_inline name= found to rewrite for {name}")
    return out


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = Path(args.pop(0))
    pub_srcs = {}
    pub_subs = {}
    shapes = "all"
    nocoop = False
    null = False
    seed = 20261030
    inputs = 1
    rounds = 120
    flags = {}
    own = set()
    modes = {}
    wsoff = {}
    phase = {}
    early = []
    sets = {}
    tkt = set()
    pick = {}
    while args:
        a = args.pop(0)
        if a == "--pub":
            name, _, path = args.pop(0).rpartition("=")
            name = name or "pub"
            if not name.startswith("pub"):
                raise SystemExit(f"public arm {name}: names start with 'pub'")
            pub_srcs[name] = Path(path).read_text()
        elif a == "--pubsub":
            name, _, rest = args.pop(0).partition("=")
            base_name, _, sub = rest.partition(":")
            old, _, new = sub.partition("=>")
            if not name.startswith("pub") or base_name not in pub_srcs:
                raise SystemExit(f"--pubsub {name}: needs a pub* name and an earlier --pub base")
            pub_subs[name] = (base_name, old.replace("\\n", "\n"), new.replace("\\n", "\n"))
        elif a == "--shapes":
            shapes = args.pop(0)
        elif a == "--nocoop":
            nocoop = True
        elif a == "--null":
            null = True
        elif a == "--seed":
            seed = int(args.pop(0))
        elif a == "--inputs":
            inputs = int(args.pop(0))
        elif a == "--rounds":
            rounds = int(args.pop(0))
        elif a == "--set":
            key, _, val = args.pop(0).partition("=")
            sets[key] = eval(val, {})
        else:
            name, _, fl = a.partition("=")
            fl, *opts = fl.split(":")
            forced = ""
            for opt in opts:
                key, _, val = opt.partition("=")
                base_name = name.removesuffix("@own")
                if key == "mode":
                    forced = val
                elif key == "wsoff":
                    wsoff[base_name] = int(val)
                    name = base_name + "@own"
                elif key == "tkt":
                    tkt.add(base_name)
                elif key == "pick":
                    pick[base_name] = int(val)
                    name = base_name + "@own"
                elif key == "early":
                    early.append(base_name)
                    name = base_name + "@own"
                elif key == "phase":
                    if not base_name.endswith("_probe"):
                        raise SystemExit(f"{base_name}: phase arms end in _probe")
                    phase[base_name] = int(val)
                    name = base_name + "@own"
                else:
                    raise SystemExit(f"unknown arm option {opt}")
            if forced:
                modes[name.removesuffix("@own")] = int(forced)
                name = name.removesuffix("@own") + "@own"
            if name.endswith("@own"):
                name = name[:-4]
                own.add(name)
            if name in ("ctrl", "ctrl2", "nocoop") or name.startswith("pub"):
                raise SystemExit(f"variant name {name} is reserved")
            flags[name] = [f for f in fl.split(",") if f]
    if null:
        flags["ctrl2"] = []
        own.add("ctrl2")
        for name in list(pub_srcs):
            if not name.startswith("pubstamp") and not name.endswith("_probe"):
                dup = name + "2"
                if dup in pub_srcs:
                    raise SystemExit(f"--null: arm {dup} already given")
                pub_srcs[dup] = _dup_pub(pub_srcs[name], f"dup_{dup}")
    src = (ROOT / "muse_attention_decode/submission.py").read_text()
    runner = base.RUNNER.format(variants=["_one"], coop_probe=True)
    # Keep the shared helpers (inputs, reference, checks); drop the self-test, stress and
    # bench calls at the end, which this bench replaces.
    head, sep, _ = runner.rpartition("\nimport os as _os\n")
    if not sep:
        raise SystemExit("runner layout changed: no 'import os as _os' block")
    config = "\n".join([
        f"_AB_FLAGS = {flags!r}",
        f"_AB_OWN = {sorted(own)!r}",
        f"_AB_PUB_SRCS = {pub_srcs!r}",
        f"_AB_PUB_SUBS = {pub_subs!r}",
        f"_AB_SHAPES = {shapes!r}",
        f"_AB_NOCOOP = {nocoop!r}",
        f"_AB_SEED = {seed!r}",
        f"_AB_INPUTS = {inputs!r}",
        f"_AB_ROUNDS = {rounds!r}",
        f"_AB_MODES = {modes!r}",
        f"_AB_WSOFF = {wsoff!r}",
        f"_AB_PHASE = {phase!r}",
        f"_AB_EARLY = {early!r}",
        f"_AB_SET = {sets!r}",
        f"_AB_TKT = {sorted(tkt)!r}",
        f"_AB_PICK = {pick!r}",
    ])
    src += head + sep + AB.replace("@@CONFIG@@", config, 1)
    out.write_text(src)
    print(f"wrote {out} with variants {list(flags)} (own state: {sorted(own)}, "
          f"forced modes: {modes})"
          f"{' + ' + str(list(pub_srcs)) if pub_srcs else ''}, shapes {shapes}, seed {seed}, "
          f"timing inputs {inputs}, rounds {rounds}")


if __name__ == "__main__":
    main()
