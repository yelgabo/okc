"""Stable prepare-time picks (archive/docs/plan-2026-10-30-tuner-stability.md, design 1) on a layer
file: the MoE, front and attention-boundary tuners time their candidates in STABLE_ROUNDS
interleaved rounds with the order rotated each round, and keep the board entry's receipt pick
(3fbb3ad027e84c6b) unless another candidate saves more than max(0.3 us, 2 x the median absolute
deviation of the preferred pick's rounds) in median paired difference and is faster in all
rounds but one. The MoE tuner keeps its one-pass screen over every variant and runs the rounds
on the variants within max(3 us, 3%) of the fastest plus the preferred one. The W4 tuner is
unchanged: its configs only matter when no fused MoE variant is picked. SUMMARY gets one
"stable" line per tuner and token count with every paired saving, its round count and the
margin. No device code changes.

--pref=KIND:T=NAME (repeatable) replaces one preferred pick, for a candidate whose own receipt
differs from the release's (KIND moe, front, abnd or m1). --m1 also moves _m1_tune (an M1 file)
to the rule: per bucket, off / o / o+front are timed as whole-step loads in STABLE_ROUNDS
rotated rounds and the preferred mode (STABLE_PREF["m1"], which --pref must then fill for
every bucket) stays unless another wins beyond the noise; the stop rule at 8 tokens is
unchanged. Without either option the output is byte-identical to the earlier tool's.

    python3 tools/dl_stable_tuner_patch.py IN.py OUT.py "HEADER LINE 2" [--pref=...] [--m1]
"""
import sys

HELPER = '''# Prepare-time picks: two copies of one file picked differently in the four-arm benches
# (724aa0dd6ed140ca, 2dc566298be1498f, 40b9b3f795f84073), widening the nulls and moving one
# load by about 1% at random. A tuner keeps the board entry's receipt pick (3fbb3ad027e84c6b)
# unless another candidate is faster by more than the noise in nearly every round
# (archive/docs/plan-2026-10-30-tuner-stability.md, design 1).
STABLE_ROUNDS = 9
STABLE_MARGIN_US = 0.3
STABLE_PREF = {
    "moe": {1: "sk8w3s", 8: "sk8w3s", 16: "sk8w3s", 32: "main-lean-grp-nohoist",
            128: "main-lean-grp-nohoist", 256: "main-lean-grp-nohoist"},
    "front": {1: "fp32", 8: "skf-r", 16: "skf-r", 32: "skf-r", 128: "concat-r", 256: "concat-r"},
    "abnd": {1: "c8x256", 8: "c1x1024", 16: "sglang", 32: "sglang", 128: "sglang",
             256: "c1x512"},
}


def _stable_pick(rounds, preferred):
    """rounds: candidate -> its time (us) in each interleaved round. Returns (pick, note). The
    preferred candidate stays unless another one's median paired saving against it exceeds
    max(STABLE_MARGIN_US, 2 x the median absolute deviation of the preferred's rounds) and it
    is faster in all rounds but one; of those, the largest saving wins. If the preferred
    candidate is not among them (dropped by a check, or not built), the lowest median wins."""
    med = {k: sorted(v)[len(v) // 2] for k, v in rounds.items()}
    if preferred not in rounds:
        best = min(med, key=med.get)
        return best, f"pref {preferred} absent, lowest median -> {best}"
    base = rounds[preferred]
    mid = med[preferred]
    mad = sorted(abs(t - mid) for t in base)[len(base) // 2]
    margin = max(STABLE_MARGIN_US, 2 * mad)
    best, gain, parts = preferred, 0.0, []
    for k, v in rounds.items():
        if k == preferred:
            continue
        d = [a - b for a, b in zip(base, v)]
        saving = sorted(d)[len(d) // 2]
        wins = sum(x > 0 for x in d)
        parts.append(f"{k} {saving:+.2f}/{wins}")
        if saving > margin and wins >= len(d) - 1 and saving > gain:
            best, gain = k, saving
    return best, (f"pref {preferred} {mid:.1f} margin {margin:.2f} rounds {len(base)} "
                  + " ".join(parts) + f" -> {best}")


'''

MOE_TUNE = '''def _moe1_tune(state, variants, seeds=(11, 12, 13, 14)):
    """Per TUNE_TOKENS count, the main-stream MoE call (its routing metadata built outside the
    timed graph, as the side stream builds it in kernel_fn) for the W4 chain and every passing
    fused variant, as cold-L2 graph replays on judge-like routings. A first pass times each
    once per routing (mean of the routings); the variants within max(3 us, 3%) of the fastest,
    plus the receipt pick, are then timed in STABLE_ROUNDS interleaved rounds (each round the
    mean over the routings) and _stable_pick decides. Only timings are kept: the choice is a
    variant per token count, fixed before any timed call."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    policy = {}
    for bs in TUNE_TOKENS:
        cands = [None] + [v for v in variants
                          if _MOE1_VARIANTS[v][1] == "skinny" or bs >= _MOE1_MAIN_MIN_M]
        tot = {v: 0.0 for v in cands}
        busiest = []
        routings = []
        for seed in seeds:
            x, tw, ids, start = _moe_judge_routing(bs, _TUNE_Q_LEN[bs], 977 * seed + bs, state)
            routings.append((x, tw, ids, start))
            loc = ids.long().flatten() - start
            busiest.append(int(torch.bincount(loc[(loc >= 0) & (loc < LOCAL_EXPERTS)],
                                              minlength=LOCAL_EXPERTS).max()))
            for v in cands:
                meta = _route_meta(tw, ids, start, bs, state, dict(_moe_opts(bs, state), fused=v))
                tot[v] += _graph_time(lambda: _experts_run(x, meta, state), flush)
        n = len(seeds)
        line = " ".join(f"{_moe1_name(v)}:{t / n:.1f}"
                        for v, t in sorted(tot.items(), key=lambda kv: kv[1]))
        print(f"MOE1BENCH T={bs} {line} us (graph replay, cold L2, mean of {n} judge-like "
              f"routings, busiest local expert {busiest} routes)")
        state.setdefault("moe1_bench", {})[bs] = {_moe1_name(v): t / n for v, t in tot.items()}
        state.setdefault("moe1_busiest", {})[bs] = busiest
        pickable = {v: t / n for v, t in tot.items() if _moe1_name(v) not in MOE1_HELD}
        fast = min(pickable.values())
        pref = STABLE_PREF["moe"].get(bs)
        short = [v for v, t in pickable.items()
                 if t <= fast + max(3.0, 0.03 * fast) or _moe1_name(v) == pref]
        if len(short) == 1:
            policy[bs] = short[0]
            state.setdefault("stable_report", {})[f"moe T={bs}"] = (
                f"one candidate within the screen -> {_moe1_name(short[0])}")
            continue
        graphs = {}
        for k, (x, tw, ids, start) in enumerate(routings):
            for v in short:
                meta = _route_meta(tw, ids, start, bs, state, dict(_moe_opts(bs, state), fused=v))
                graphs[(k, v)] = _graph_build(lambda x=x, meta=meta: _experts_run(x, meta, state))
        rounds = {_moe1_name(v): [] for v in short}
        for r in range(STABLE_ROUNDS):
            acc = {v: 0.0 for v in short}
            for k in range(len(routings)):
                for v in short[r % len(short):] + short[:r % len(short)]:
                    acc[v] += _graph_median(graphs[(k, v)], flush, 10)
            for v in short:
                rounds[_moe1_name(v)].append(acc[v] / len(routings))
        del graphs
        pick, note = _stable_pick(rounds, pref)
        policy[bs] = next(v for v in short if _moe1_name(v) == pick)
        state.setdefault("stable_report", {})[f"moe T={bs}"] = note
    clean, msg = _moe1_check(state)
    if not clean:
        print(f"MOE1BENCH scratch dirty after the bench ({msg}); fused policy discarded")
        _FAILED.append(f"moe1 bench: scratch dirty ({msg}), policy discarded")
        return {}
    print(f"MOE1POLICY {{{', '.join(f'{b}: {_moe1_name(v)}' for b, v in policy.items())}}}")
    return policy


'''

FRONT_TUNE = '''def _front_tune(state):
    """Times the MoE front per token count, as CUDA graphs replayed after an L2 flush as the
    judge times it, for every side-chain variant, in STABLE_ROUNDS interleaved rounds, and
    picks with _stable_pick (the receipt's front unless another is faster beyond the noise).
    The variants must agree with the reference ("split" above 8 tokens, "fp32" up to 8): the
    normalized expert input within 1e-3 (relative L2), the level of BF16 rounding flips from a
    different FP32 summation order, and at least 99% of the routes equal (as sets per token); a
    variant that disagrees is dropped at that token count. Up to 8 tokens the FP32 router input
    is the BF16 input widened, so the variants route the same values.

    Where a route variant (ROUTE_VARIANTS) is offered, the rounds time the front and the MoE
    together, since the route variants' saving includes the MoE's PDL start; the front-only
    times (one median each) are printed beside them. A route variant must also match its base
    variant (same logits): equal top-8 sets, the normalized input within 1e-3, the MoE output
    within 1e-3, and after ROUTE_STRESS more replays of its graph the same output and a clean
    MoE scratch (no spin timeout)."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(21)
    local_start = torch.tensor([96], dtype=torch.int32, device="cuda")
    policy = {}
    report = state.setdefault("front_report", {})
    for tokens in FRONT_TOKENS:
        variants = _front_variants(tokens, state)
        if len(variants) == 1:
            policy[tokens] = variants[0]
            continue
        ffn_input = torch.randn((tokens, HIDDEN), generator=gen, device="cuda")
        ffn_input = ffn_input.to(torch.bfloat16)
        router_input = ffn_input.float() if tokens <= ROUTER_FP32_MAX_TOKENS else None
        opts = _moe_opts(tokens, state) if state["moe_ours"] else None
        with_moe = any(v in ROUTE_VARIANTS for v in variants)
        outs, graphs = {}, {}
        for variant in variants:

            def front(res, variant=variant):
                ei, tw, ids, meta = _moe_front(ffn_input, router_input, local_start, state,
                                               opts, variant)
                res["x"] = ei if meta is not None else _pre_norm(ei, state)
                res["ids"] = ids
                return meta, tw

            def front_moe(res):
                meta, tw = front(res)
                if meta is not None:
                    res["y"] = _experts_run(res["x"], meta, state)
                else:
                    res["y"] = _experts(res["x"], tw, res["ids"], local_start, state, opts)

            fres, cres = {}, {}
            fgraph = _graph_build(lambda: front(fres))
            fgraph.replay()
            torch.cuda.synchronize()
            outs[variant] = [fres["x"].clone(), fres["ids"].clone(), None]
            cgraph = None
            if with_moe:
                cgraph = _graph_build(lambda: front_moe(cres))
                cgraph.replay()
                torch.cuda.synchronize()
                if cres.get("y") is not None:
                    outs[variant][2] = cres["y"].clone()
                if variant in ROUTE_VARIANTS and cres.get("y") is not None:
                    for _ in range(ROUTE_STRESS):
                        cgraph.replay()
                    torch.cuda.synchronize()
                    outs[variant].append(cres["y"].clone())
            graphs[variant] = (fgraph, cgraph)
        reference = variants[0]
        ref_x, ref_ids = outs[reference][:2]
        agree, alive = {}, []
        for variant, (x, ids, *_rest) in outs.items():
            err = _rel(x, ref_x)
            same = float((ids.long().sort(-1).values == ref_ids.long().sort(-1).values)
                         .float().mean())
            agree[variant] = f"{err:.1e}/{same:.3f}"
            bad = None
            if not (err < 1e-3 and same >= 0.99):
                bad = f"rel diff {err:.2e}, routes {same:.3f} against {reference}"
            if variant in ROUTE_VARIANTS and bad is None:
                bad = _route_variant_check(outs[variant], outs.get(ROUTE_VARIANTS[variant]),
                                           state)
            if bad is not None:
                print(f"TUNE front T={tokens} {variant} dropped: {bad}")
                _FAILED.append(f"front {variant} T={tokens} ({bad[:80]})")
            else:
                alive.append(variant)
        rounds = {v: [] for v in alive}
        for r in range(STABLE_ROUNDS):
            for v in alive[r % len(alive):] + alive[:r % len(alive)]:
                rounds[v].append(_graph_median(graphs[v][1 if with_moe else 0], flush, 20))
        med = {v: sorted(t)[len(t) // 2] for v, t in rounds.items()}
        if with_moe:
            chain = med
            times = {v: _graph_median(graphs[v][0], flush, 20) for v in alive}
        else:
            chain, times = {}, med
        del graphs
        best, note = _stable_pick(rounds, STABLE_PREF["front"].get(tokens))
        state.setdefault("stable_report", {})[f"front T={tokens}"] = note
        policy[tokens] = best
        report[tokens] = (dict(times), dict(chain))
        line = (f"TUNE front T={tokens} "
                + " ".join(f"{v}:{t:.1f}" for v, t in sorted(times.items(), key=lambda kv: kv[1])))
        if with_moe:
            line += (" | +moe " + " ".join(f"{v}:{t:.1f}" for v, t in
                                           sorted(chain.items(), key=lambda kv: kv[1])))
        print(line + f" -> {best} (vs {reference}: "
              + " ".join(f"{v} {a}" for v, a in agree.items()) + ")")
    state["front_policy"] = policy


'''

ABND_REPS = [
    ('''    L2 flush, in six rounds with the order rotated each round, and keeps the candidate whose
    median over the six round medians is lowest.''',
     '''    L2 flush, in STABLE_ROUNDS rounds with the order rotated each round, and picks with
    _stable_pick (the receipt's boundary unless another is faster beyond the noise).'''),
    ('''        for r in range(6):
            for name in order[r % len(order):] + order[:r % len(order)]:''',
     '''        for r in range(STABLE_ROUNDS):
            for name in order[r % len(order):] + order[:r % len(order)]:'''),
    ('''        best = min(times, key=times.get)
        incumbent = ABND_GEOS[0]
        if incumbent in times and times[best] > times[incumbent] - ABND_MARGIN_US:
            best = incumbent
''',
     '''        best, note = _stable_pick(rounds, STABLE_PREF["abnd"].get(tokens))
        state.setdefault("stable_report", {})[f"abnd T={tokens}"] = note
'''),
]

SUMMARY_OLD = '''    lines.append(f"failed {len(_FAILED)}: "'''
SUMMARY_NEW = '''    for key, note in state.get("stable_report", {}).items():
        lines.append(f"stable {key}: {note}")
    lines.append(f"failed {len(_FAILED)}: "'''


def cut(text, start, end, new):
    a = text.index(start)
    b = text.index(end, a)
    return text[:a] + new + text[b:]


# A file built by tools/dl_gate_patch.py checks route variants with the fresh-input gate
# instead of replaying the timing graph; the stable front tuner keeps whichever check the file
# has.
GATED_FRONT = [
    ('''    within 1e-3, and after ROUTE_STRESS more replays of its graph the same output and a clean
    MoE scratch (no spin timeout)."""''',
     '''    within 1e-3, the fresh-input replay gate (_route_fresh_gate, ROUTE_STRESS replays against
    as many eager calls) and a clean MoE scratch (no spin timeout)."""'''),
    ('''                if variant in ROUTE_VARIANTS and cres.get("y") is not None:
                    for _ in range(ROUTE_STRESS):
                        cgraph.replay()
                    torch.cuda.synchronize()
                    outs[variant].append(cres["y"].clone())''',
     '''                if variant in ROUTE_VARIANTS and cres.get("y") is not None:
                    outs[variant].append(_route_fresh_gate(
                        lambda: front_moe(cres), cres, ffn_input, router_input,
                        f"{variant}@{tokens}", state))'''),
]
FRONT_ENDS = ("# Extra replays of a route variant's front + MoE graph in the tuner",
              "# Replays of a route variant's front + MoE graph in the tuner under the")

M1_OLD = '''        med = _m1_pair(state, graphs, flush, 9)
        del graphs
        load = {mode: (46 * med[(mode, "sliding")] + 16 * med[(mode, "full")]) / 62
                for mode in ("off", "o", "o+front")}
        best = min(load, key=load.get)
'''
M1_NEW = '''        per = _m1_rounds(graphs, flush)
        del graphs
        med = {k: sorted(v)[len(v) // 2] for k, v in per.items()}
        rounds = {mode: [(46 * s + 16 * f) / 62 for s, f in zip(per[(mode, "sliding")],
                                                               per[(mode, "full")])]
                  for mode in ("off", "o", "o+front")}
        load = {mode: sorted(v)[len(v) // 2] for mode, v in rounds.items()}
        best, note = _stable_pick(rounds, STABLE_PREF["m1"].get(bucket))
        state.setdefault("stable_report", {})[f"m1 T={bucket}"] = note
'''
M1_ROUNDS = '''def _m1_rounds(graphs, flush):
    """STABLE_ROUNDS rounds over M1's modes, the mode order rotated each round and both layers
    of a mode back to back; (mode, layer) -> its cold-L2 graph median (us) in each round."""
    modes = list(dict.fromkeys(m for m, _ in graphs))
    times = {k: [] for k in graphs}
    for r in range(STABLE_ROUNDS):
        for m in modes[r % len(modes):] + modes[:r % len(modes)]:
            for k in graphs:
                if k[0] == m:
                    times[k].append(_graph_median(graphs[k], flush, 10))
    return times


def _m1_tune(state):'''

DEFAULT_PREF = {
    "moe": {1: "sk8w3s", 8: "sk8w3s", 16: "sk8w3s", 32: "main-lean-grp-nohoist",
            128: "main-lean-grp-nohoist", 256: "main-lean-grp-nohoist"},
    "front": {1: "fp32", 8: "skf-r", 16: "skf-r", 32: "skf-r", 128: "concat-r", 256: "concat-r"},
    "abnd": {1: "c8x256", 8: "c1x1024", 16: "sglang", 32: "sglang", 128: "sglang",
             256: "c1x512"},
}


def pref_block(pref):
    lines = ["STABLE_PREF = {"]
    for kind, table in pref.items():
        lines.append(f'    "{kind}": {{')
        lines += [f'        {t}: "{name}",' for t, name in table.items()]
        lines.append("    },")
    return "\n".join(lines) + "\n}\n"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 3:
        raise SystemExit(__doc__)
    src, out, header = args
    pref = {k: dict(v) for k, v in DEFAULT_PREF.items()}
    m1 = False
    for a in sys.argv[1:]:
        if a == "--m1":
            m1 = True
        elif a.startswith("--pref="):
            kind, _, rest = a[len("--pref="):].partition(":")
            t, _, name = rest.partition("=")
            if kind not in ("moe", "front", "abnd", "m1") or not t.isdigit() or not name:
                raise SystemExit(f"bad {a}")
            pref.setdefault(kind, {})[int(t)] = name
        elif a.startswith("--"):
            raise SystemExit(f"unknown option {a}")
    if m1 and sorted(pref.get("m1", {})) != [8, 16, 32]:
        raise SystemExit("--m1 needs --pref=m1:8=..., --pref=m1:16=... and --pref=m1:32=...")
    if "m1" in pref and not m1:
        raise SystemExit("--pref=m1:... needs --m1")
    helper = HELPER
    if pref != DEFAULT_PREF:
        a = helper.index("STABLE_PREF = {")
        b = helper.index("\n}\n", a) + 3
        helper = helper[:a] + pref_block(pref) + helper[b:]
    text = open(src).read()
    if "STABLE_ROUNDS" in text:
        raise SystemExit("IN already has the stable tuners")
    for start in ("def _moe1_tune(state, variants", "def _front_tune(state):",
                  "def _moe1_enable(state, weights):"):
        if text.count(start) != 1:
            raise SystemExit(f"expected one {start!r}")
    ends = [e for e in FRONT_ENDS if text.count(e) == 1]
    if len(ends) != 1:
        raise SystemExit("the comment after _front_tune is not where expected")
    front = FRONT_TUNE
    if ends[0] == FRONT_ENDS[1]:
        if "def _route_fresh_gate(" not in text:
            raise SystemExit("gated front comment without _route_fresh_gate")
        for old, new in GATED_FRONT:
            if front.count(old) != 1:
                raise SystemExit(f"FRONT_TUNE changed: {old[:60]!r}")
            front = front.replace(old, new)
    text = cut(text, "def _moe1_tune(state, variants", "def _moe1_enable(state, weights):",
               helper + MOE_TUNE)
    text = cut(text, "def _front_tune(state):", ends[0], front)
    reps = ABND_REPS + [(SUMMARY_OLD, SUMMARY_NEW)]
    if m1:
        reps += [(M1_OLD, M1_NEW), ("def _m1_tune(state):", M1_ROUNDS),
                 ("    off, o and o + front per bucket; the fastest load wins per bucket. Stop",
                  "    off, o and o + front per bucket; _stable_pick decides per bucket. Stop")]
    for old, new in reps:
        if text.count(old) != 1:
            raise SystemExit(f"not where expected: {old[:60]!r}")
        text = text.replace(old, new)
    lines = text.split("\n")
    if lines[0] != "# okc: problem=muse_decode_layer":
        raise SystemExit("line 1 is not the okc header")
    lines[1] = header
    open(out, "w").write("\n".join(lines))
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
