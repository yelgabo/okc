"""Prespecified full30 score gate for the owned SK sparse-repair experiment.

Raw paired event factors and common-offset sensitivities are not official
candidate scores. Twenty unchanged operators use actual common measurements;
their samples are shared across comparisons and supply no independent null.
"""
import math
import statistics

LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")
M_VALUES = (1, 8, 16, 32, 128, 256)
SEEDS = (903151, 903157, 903163)
ROUNDS = 24
TARGETS = frozenset({("qkv_gate", 8), ("qkv_gate", 16), ("qkv_gate", 32),
    ("o", 16), ("o", 32), ("pooler", 16), ("pooler", 32),
    ("unpooler", 8), ("unpooler", 16), ("unpooler", 32)})


def score_model(rows, offsets=None):
    """Evaluate all rows using six equally weighted sums of five operators.

    Each official cell has weight62, which cancels within its load sum. A
    non-None offset map applies the SAME per-cell raw-minus-adjusted offset
    to every arm. Only the owning terminal receipt may supply that map.
    """
    expected = {(layer, m, seed) for layer in LAYERS for m in M_VALUES for seed in SEEDS}
    if len(rows) != len(expected):
        raise ValueError("requires all ninety prespecified rows")
    lookup = {}
    for row in rows:
        key = (row["layer"], row["m"], row["seed"])
        if key not in expected or key in lookup:
            raise ValueError("duplicate or unexpected projection row")
        target = key[:2] in TARGETS
        if row["target"] is not target:
            raise ValueError("target metadata does not match frozen ten cells")
        arms = ("c0", "c1", "t0", "t1", "anchor") if target else ("common",)
        if set(row["samples_us"]) != set(arms):
            raise ValueError("incomplete arm samples")
        times = {}
        for arm in arms:
            samples = row["samples_us"][arm]
            if len(samples) != ROUNDS or not all(isinstance(t, (int, float)) and
                    not isinstance(t, bool) and math.isfinite(t) and t > 0 for t in samples):
                raise ValueError("invalid or incomplete raw event vector")
            times[arm] = statistics.median(samples)
        lookup[key] = times
    if set(lookup) != expected:
        raise ValueError("missing projection row")
    if offsets is not None and (set(offsets) != {k[:2] for k in expected} or
            not all(math.isfinite(v) for v in offsets.values())):
        raise ValueError("requires all thirty finite contemporary offsets")

    def latency(layer, m, seed, arm):
        key = (layer, m, seed)
        t = lookup[key][arm if (layer, m) in TARGETS else "common"]
        adjusted = t - (offsets[(layer, m)] if offsets is not None else 0.)
        if not math.isfinite(adjusted) or adjusted <= 0:
            raise ValueError("nonpositive offset-adjusted latency")
        return adjusted

    def factor(seed, baseline, treatment):
        logs = []
        for m in M_VALUES:
            before = sum(latency(layer, m, seed, baseline) for layer in LAYERS)
            after = sum(latency(layer, m, seed, treatment) for layer in LAYERS)
            logs.append(math.log(before / after))
        return math.exp(sum(logs) / len(M_VALUES))

    null_log = max(abs(math.log(factor(seed, "c0", "c1"))) for seed in SEEDS)
    regressions, anchor_drift = [], []
    per_seed = []
    for seed in SEEDS:
        crosses = {t + "/" + c: factor(seed, c, t)
                   for t in ("t0", "t1") for c in ("c0", "c1", "anchor")}
        for layer, m in sorted(TARGETS):
            anchor = latency(layer, m, seed, "anchor")
            for c in ("c0", "c1"):
                delta = abs(latency(layer, m, seed, c) - anchor)
                if delta > max(.5, .01 * anchor):
                    anchor_drift.append([layer, m, seed, c, delta])
            for t in ("t0", "t1"):
                for c in ("c0", "c1", "anchor"):
                    base = latency(layer, m, seed, c)
                    delta = latency(layer, m, seed, t) - base
                    if delta > max(.5, .01 * base):
                        regressions.append([layer, m, seed, t, c, delta])
        keep = all(v >= 1.01 and math.log(v) > 2 * null_log for v in crosses.values())
        per_seed.append({"seed": seed, "cross_factors": crosses, "factor_gate": keep})
    return {"per_seed": per_seed, "null_log": null_log,
            "minimum_factor": min(v for row in per_seed for v in row["cross_factors"].values()),
            "anchor_drift": anchor_drift, "regressions": regressions,
            "keep": all(row["factor_gate"] for row in per_seed) and not anchor_drift and not regressions,
            "scope": "Full30 paired event factors; unchanged20 share actual common samples, not independent null; not official candidate score"}
