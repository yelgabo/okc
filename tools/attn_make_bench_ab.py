"""Write a paired A/B bench for the one-launch kernel: submission.py (the judged path is the
control) plus extra builds of attn_one.cu with compile flags, and optionally the public entry's
kernel loaded from its own file, all timed in one process on the same inputs.

    python3 tools/attn_make_bench_ab.py OUT.py [--pub [NAME=]PUBLIC.py ...]
        [--shapes sliding|full|all] NAME=FLAGS ...

NAME=FLAGS adds a variant built from submission.py's inlined _ONE_SRC (not attn_one.cu) with
those nvcc flags (comma-separated), e.g. diet=-DONE_DIAG=0,-DONE_DIET=1. A switch that exists
only in attn_one.cu needs `tools/attn_inline_one.py OUT.py` on the written bench. The variant
runs through _one with its module swapped in, so it shares the control's plan and workspace.

At import the bench prints, before the judge's own tests and timing:

  ABCHECK   per variant and shape (scored shapes and seven tp8 test shapes): for the control
            the judge-style check against the FP32 reference, for every other variant its
            output against the control's (relative L2 below 2e-2, differing elements), and the
            workspace invariant after the call.
  ABCOLD    per scored shape: median of 120 graph replays after a 256 MB L2 flush, the
            variants interleaved in a rotating order within each round.
  ABWARM    per scored shape: a graph of 16 back-to-back calls, median over 25 replays,
            divided by 16 (L2 warm, launch gaps of a captured graph).

Arms whose name ends in `_probe` are timing probes with wrong or unguaranteed output: ABCHECK
prints PROBE for them and does not count them as failures. An arm whose name starts with `stamp`
(built with -DONE_STAMP=1) also gets STAMP and STAMPWAVE lines: per-CTA records of five cold calls per full-context
q_len 8 shape (attn_one.cu, ONE_STAMP). `--nocoop` adds the arm `nocoop`: the control's module
launched without cudaLaunchAttributeCooperative, same grid.

A public arm whose name starts with
`pubstamp` (a copy of the public file with globaltimer stamps written to dbg[8192 + 8 c ..]) gets
PUBSTAMP and STAMPWAVE lines: the same loop-end spread by dispatch wave and by SM.

--pub may repeat; each public file is an arm (default name `pub`; its names must start with
`pub`, and a second file needs its own load_inline name). --shapes full keeps the full-context
shapes with batch > 1 (modes 49 and 51). The public files are read at generation time and embedded as a string; the output belongs in the
session scratchpad, not in the repository (only the logs of its runs are committed).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import attn_make_bench_one as base  # noqa: E402

AB = r'''

_AB_FLAGS = {flags!r}
_AB_PUB_SRCS = {pub_srcs!r}
_AB_SHAPES = {shapes!r}
_AB_NOCOOP = {nocoop!r}
_AB_MODS = {{}}
for _name, _fl in _AB_FLAGS.items():
    try:
        _AB_MODS[_name] = load_inline(
            name=f"okc_attn_one_ab_{{_name}}",
            cpp_sources=_ONE_CPP,
            cuda_sources=_ONE_SRC,
            functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", *_fl],
        )
        print(f"ABBUILD {{_name}} {{_fl}} loaded {{_ob_elapsed()}}", flush=True)
    except Exception as _exc:
        print(f"ABBUILD {{_name}} FAIL {{type(_exc).__name__}}: {{str(_exc)[:400]}}", flush=True)
_AB_PUBS = {{}}
for _name, _src in _AB_PUB_SRCS.items():
    try:
        import types as _types
        _m = _types.ModuleType(f"okc_ab_{{_name}}")
        exec(compile(_src, f"{{_name}}_attention.py", "exec"), _m.__dict__)
        _m._module()
        _AB_PUBS[_name] = _m
        print(f"ABBUILD {{_name}} loaded {{_ob_elapsed()}}", flush=True)
    except Exception as _exc:
        print(f"ABBUILD {{_name}} FAIL {{type(_exc).__name__}}: {{str(_exc)[:400]}}", flush=True)


def _ab_via(mod):
    def call(data):
        global _ONE
        saved = _ONE
        _ONE = mod
        try:
            return _one(data)
        finally:
            _ONE = saved
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
    fns = {{"ctrl": _one}}
    if _AB_NOCOOP:
        fns["nocoop"] = _ab_nocoop
    for name, mod in _AB_MODS.items():
        fns[name] = _ab_via(mod)
    for name, mod in _AB_PUBS.items():
        fns[name] = mod.kernel_fn
    return fns


def _ab_scored():
    for batch, layer, q_len in _OB_SCORED:
        if _AB_SHAPES == "sliding" and layer != "sliding":
            continue
        if _AB_SHAPES == "full" and (layer == "sliding" or batch == 1):
            continue
        yield batch, layer, q_len


_AB_TALLY = {{}}


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
        name = "b{{}} ctx{{}} {{}} {{}} L{{}} {{}} {{}} page{{}} scale{{:g}}".format(*shape)
        try:
            data = _ob_inputs(*shape)
            if not _one_ok(data["config"]):
                continue
            plan = _ob_plan_str(data)
            ref = None
            for vname, fn in fns.items():
                out = _ob_run(lambda: fn(data), f"{{vname}} {{name}}")
                torch.cuda.synchronize()
                wok, wmsg = _ob_ws_check(dev) if not vname.startswith("pub") else (True, "")
                # The judge-style check (two FP32 references) for the control only; the
                # variants are compared with the control's output.
                if ref is None:
                    ok, msg = _ob_check(data, out)
                    ref = out.float().clone()
                    vs = ""
                else:
                    rel = _ob_rel(out, ref)
                    ok, msg = rel < 2e-2, ""
                    vs = f"vs_ctrl={{rel:.1e}} {{_ob_diffstat(out, ref)}}"
                status = "PASS" if ok and wok else "FAIL"
                if vname.endswith("_probe"):
                    status = "PROBE"
                fails += status == "FAIL"
                tally = _AB_TALLY.setdefault(vname, {{"PASS": 0, "FAIL": 0, "PROBE": 0}})
                tally[status] += 1
                print(f"ABCHECK {{status}} {{vname}} {{name}} [{{plan}}] {{msg}}{{vs}} {{wmsg}}",
                      flush=True)
            data = None
        except Exception as exc:
            fails += 1
            print(f"ABCHECK FAIL {{name}} {{type(exc).__name__}}: {{str(exc)[:300]}}", flush=True)
        torch.cuda.empty_cache()
    print(f"ABCHECK done: {{fails}} failing {{_ob_elapsed()}}", flush=True)


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
    return xs[len(xs) // 2]


def _ab_bench(rounds=120, warm_calls=16, warm_reps=25):
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    fns = _ab_fns()
    names = list(fns)
    for batch, layer, q_len in _ab_scored():
        data = _ob_inputs(batch, 80000, "tp8", layer, q_len, "fixed" if batch == 1 else "agentic",
                          "soft" if layer == "sliding" else "coherent", 1, 1.0)
        tag = f"b={{batch:2d}} {{layer:7s}} L={{q_len}}"
        try:
            cold = {{n: _ab_graph(fns[n], data, 1) for n in names}}
            samples = {{n: [] for n in names}}
            for r in range(rounds):
                order = names[r % len(names):] + names[:r % len(names)]
                evs = []
                for n in order:
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
            row = " ".join(f"{{n}}={{_ab_med(samples[n]):.2f}}" for n in names)
            print(f"ABCOLD {{tag}} {{row}} {{_ob_elapsed()}}", flush=True)
            warm = {{n: _ab_graph(fns[n], data, warm_calls) for n in names}}
            ws = {{n: [] for n in names}}
            for r in range(warm_reps):
                order = names[r % len(names):] + names[:r % len(names)]
                evs = []
                for n in order:
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
            row = " ".join(f"{{n}}={{_ab_med(ws[n]):.2f}}" for n in names)
            print(f"ABWARM {{tag}} {{row}} {{_ob_elapsed()}}", flush=True)
        except Exception as exc:
            print(f"ABBENCH FAIL {{tag}} {{type(exc).__name__}}: {{str(exc)[:300]}}", flush=True)
        data = None
        torch.cuda.empty_cache()


_ST_BASE, _ST_W = 1024, 16


def _ab_wave(tag, name, rep, ends, sms):
    """Loop-end spread by dispatch wave and by SM: ends[c] in us for CTA c, sms[c] its SM."""
    n = len(ends)
    h = n // 2

    def q(xs):
        xs = sorted(xs)
        if not xs:
            return "-"
        return "{{:.1f}}/{{:.1f}}/{{:.1f}}".format(xs[0], xs[len(xs) // 2], xs[-1])
    by_sm = {{}}
    for c in range(n):
        by_sm.setdefault(int(sms[c]), []).append(c)
    pairs = [sorted(v) for v in by_sm.values() if len(v) == 2]
    gaps = [ends[b] - ends[a] for a, b in pairs]
    low_first = sum(1 for g in gaps if g > 0)
    sm_end = [max(ends[c] for c in v) for v in by_sm.values()]
    print(f"STAMPWAVE {{tag}} {{name}} rep={{rep}} ctas={{n}} sms={{len(by_sm)}} pairs={{len(pairs)}} "
          f"wave1_end={{q(ends[:h])}} wave2_end={{q(ends[h:])}} "
          f"pair_gap_hi_minus_lo={{q(gaps)}} abs_gap={{q([abs(g) for g in gaps])}} "
          f"lo_first={{low_first}}/{{len(pairs)}} sm_end={{q(sm_end)}} "
          f"mean_end={{sum(ends) / max(n, 1):.1f}} max_end={{max(ends):.1f}}", flush=True)


def _ab_pubstamps(reps=5):
    """Per-CTA records of the public flat kernel's stamp copy (arms named pubstamp*): dbg
    int64 [8192 + 8 c + k]: 0 start, 1 consumer loop end, 2 finalize done (globaltimer ns),
    3 SM id, 4 tiles, 5 segments."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for name, mod in _AB_PUBS.items():
        if not name.startswith("pubstamp"):
            continue
        for batch, layer, q_len in _ab_scored():
            if layer == "sliding" or q_len != 8:
                continue
            data = _ob_inputs(batch, 80000, "tp8", layer, q_len, "agentic", "coherent", 1, 1.0)
            tag = f"b={{batch:2d}} {{layer:7s}} L={{q_len}}"
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
                ends = [(int(x) - ref) / 1000.0 for x in rec[:, 1]]
                fin = [(int(x) - ref) / 1000.0 for x in rec[:, 2]]

                def q(xs):
                    xs = sorted(xs)
                    return "{{:.1f}}/{{:.1f}}/{{:.1f}}".format(xs[0], xs[len(xs) // 2], xs[-1])
                print(f"PUBSTAMP {{tag}} {{name}} rep={{r}} ctas={{n}} span={{max(fin):.1f}} "
                      f"start={{q(start)}} loop_end={{q(ends)}} "
                      f"barrier_fin={{q([f - e for f, e in zip(fin, ends)])}} "
                      f"tiles={{q([int(x) for x in rec[:, 4]])}} segs={{q([int(x) for x in rec[:, 5]])}} "
                      f"{{_ob_elapsed()}}", flush=True)
                _ab_wave(tag, name, r, ends, [int(x) for x in rec[:, 3]])
                if r == 0:
                    order = sorted(range(n), key=lambda i: ends[i])
                    pick = order[:4] + order[n // 2 - 2:n // 2 + 2] + order[-4:]
                    print(f"PUBSTAMPCTA {{tag}} {{name}} " + " ".join(
                        "c{{}}:sm{{}}:t{{}}:end{{:.1f}}".format(i, int(rec[i, 3]), int(rec[i, 4]), ends[i])
                        for i in pick), flush=True)
            data = None
            torch.cuda.empty_cache()


def _ab_stamps(reps=5):
    """Per-CTA records of the ONE_STAMP build on the full-context q_len 8 shapes, cold calls.
    Fields: 0 start, 1 partition done, 2 first tile ready, 3 tile loops done, 4 grid barrier
    passed, 5 finalize done (globaltimer ns, low 32 bits); 6 tiles, 7 segments, 8 consumer
    full-barrier wait, 9 segment prologues, 10 tile loops, 11 producer empty-barrier wait,
    12 producer lifetime, 13 / 14 consumer cycles to loop end / to kernel end; 15 SM id."""
    global _ONE
    for sname, mod in _AB_MODS.items():
        if sname.startswith("stamp"):
            _ab_stamps_one(sname, mod, reps)


def _ab_stamps_one(sname, mod, reps):
    global _ONE
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for batch, layer, q_len in _ab_scored():
        if layer == "sliding" or q_len != 8:
            continue
        data = _ob_inputs(batch, 80000, "tp8", layer, q_len, "agentic", "coherent", 1, 1.0)
        tag = f"b={{batch:2d}} {{layer:7s}} L={{q_len}}"
        for r in range(reps):
            buf = torch.zeros(_ST_BASE + 512 * _ST_W, dtype=torch.int32, device="cuda")
            flush.zero_()
            saved = _ONE
            _ONE = mod
            try:
                _one(data, dbg=buf)
                n = _one_plan(data["config"], data["page_table"].shape[1],
                              _one_state(data["qkv"].device)["resident"])[0]
            finally:
                _ONE = saved
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
                return "{{:.1f}}/{{:.1f}}/{{:.1f}}".format(xs[0], xs[len(xs) // 2], xs[-1])
            _ab_wave(tag, sname, r, [float(x) for x in ns[:, 3]], [int(x) for x in rec[:, 15]])
            print(f"STAMP {{tag}} {{sname}} rep={{r}} ctas={{n}} ghz={{ghz:.3f}} "
                  f"span={{float(ns[:, 5].max()):.1f}} start={{q(ns[:, 0])}} "
                  f"part={{q(ns[:, 1] - ns[:, 0])}} first_tile={{q(ns[:, 2] - ns[:, 0])}} "
                  f"loop_end={{q(ns[:, 3])}} barrier_wait={{q(ns[:, 4] - ns[:, 3])}} "
                  f"fin={{q(ns[:, 5] - ns[:, 4])}} tiles={{q(rec[:, 6])}} segs={{q(rec[:, 7])}} "
                  f"fullwait_us={{q(us[:, 0])}} prologue_us={{q(us[:, 1])}} "
                  f"loop_us={{q(us[:, 2])}} prod_emptywait_us={{q(us[:, 3])}} "
                  f"prod_life_us={{q(us[:, 4])}} {{_ob_elapsed()}}", flush=True)
            if r == 0:
                # Loop-end time next to tiles, so imbalance can be told from per-tile speed.
                order = sorted(range(n), key=lambda i: float(ns[i, 3]))
                pick = order[:4] + order[n // 2 - 2:n // 2 + 2] + order[-4:]
                print(f"STAMPCTA {{tag}} {{sname}} " + " ".join(
                    "c{{}}:sm{{}}:t{{}}:s{{}}:end{{:.1f}}:pro{{:.1f}}:fw{{:.1f}}".format(
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
            print(f"ABCHECK failed: {{type(_exc).__name__}}: {{_exc}}", flush=True)
        try:
            _ab_stamps()
        except Exception as _exc:
            print(f"STAMP failed: {{type(_exc).__name__}}: {{_exc}}", flush=True)
        try:
            _ab_pubstamps()
        except Exception as _exc:
            print(f"PUBSTAMP failed: {{type(_exc).__name__}}: {{_exc}}", flush=True)
        try:
            _ab_bench()
        except Exception as _exc:
            print(f"ABBENCH failed: {{type(_exc).__name__}}: {{_exc}}", flush=True)
        # Repeated last: the run log keeps only its final 16,000 characters.
        print("ABCHECKSUM " + " ".join(f"{{n}}={{t['PASS']}}/{{t['FAIL']}}/{{t['PROBE']}}"
                                       for n, t in _AB_TALLY.items()) + " (pass/fail/probe)",
              flush=True)
'''


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = Path(args.pop(0))
    pub_srcs = {}
    shapes = "all"
    nocoop = False
    flags = {}
    while args:
        a = args.pop(0)
        if a == "--pub":
            name, _, path = args.pop(0).rpartition("=")
            name = name or "pub"
            if not name.startswith("pub"):
                raise SystemExit(f"public arm {name}: names start with 'pub'")
            pub_srcs[name] = Path(path).read_text()
        elif a == "--shapes":
            shapes = args.pop(0)
        elif a == "--nocoop":
            nocoop = True
        else:
            name, _, fl = a.partition("=")
            flags[name] = [f for f in fl.split(",") if f]
    src = (ROOT / "muse_attention_decode/submission.py").read_text()
    runner = base.RUNNER.format(variants=["_one"], coop_probe=True)
    # Keep the shared helpers (inputs, reference, checks); drop the self-test, stress and
    # bench calls at the end, which this bench replaces.
    head, sep, _ = runner.rpartition("\nimport os as _os\n")
    if not sep:
        raise SystemExit("runner layout changed: no 'import os as _os' block")
    src += head + sep + AB.format(flags=flags, pub_srcs=pub_srcs, shapes=shapes, nocoop=nocoop)
    out.write_text(src)
    print(f"wrote {out} with variants {list(flags)}{' + ' + str(list(pub_srcs)) if pub_srcs else ''}, shapes {shapes}")


if __name__ == "__main__":
    main()
