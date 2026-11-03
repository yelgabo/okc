"""What-if model of the muse_decode_layer score from a saved run JSON.

Reads a run's per-shape medians (ours, SGLang's and the floor), prints the score the judge's
formula gives them (geomean over loads of weighted SGLang time over ours), the score gain of
10 us saved on each shape, and the score after the savings given on the command line.

    python3 tools/dl_score_model.py muse_decode_layer/profiles/submission.<job>.json
    python3 tools/dl_score_model.py <run.json> --save d1s=10 v1s=8 --save v32f=90
    python3 tools/dl_score_model.py <run.json> --target 2.14

Shape names: d1s, d1f, d16s, d16f, d32s, d32f, v1s, v1f, v16s, v16f, v32s, v32f (decode or
verify, batch, sliding or full). --target prints the uniform time cut that would reach it.
"""

import argparse
import math

from hotspots import load

LOADS = ("d1", "d16", "d32", "v1", "v16", "v32")


def load_run(path):
    d = load(path)
    if d.get("problem") != "muse_decode_layer":
        raise ValueError("expected a muse_decode_layer receipt")
    shapes = {}
    for b in d["benchmarks"]:
        case = b["case"]
        kind = "d" if case.get("q_len", 1) == 1 else "v"
        name = f"{kind}{case['batch']}{'s' if case['layer'] == 'sliding' else 'f'}"
        if name in shapes:
            raise ValueError(f"duplicate scored shape: {name}")
        shapes[name] = {
            "sglang": b["baseline_ns"] / 1000, "ours": b["median_ns"] / 1000,
            "floor": b["speed_of_light_ns"] / 1000, "weight": case["weight"],
        }
    expected = {load + x for load in LOADS for x in "sf"}
    if shapes.keys() != expected:
        raise ValueError(f"expected twelve scored shapes; missing {expected - shapes.keys()}")
    return d["job_id"], d["summary"]["speedup"], shapes


def score(shapes, ours=None):
    ours = ours or {n: s["ours"] for n, s in shapes.items()}
    logs = []
    for load in LOADS:
        base = sum(shapes[load + x]["weight"] * shapes[load + x]["sglang"] for x in "sf")
        mine = sum(shapes[load + x]["weight"] * ours[load + x] for x in "sf")
        logs.append(math.log(base / mine))
    return math.exp(sum(logs) / len(logs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--save", nargs="*", action="extend", default=[],
                    help="shape=us saved, e.g. d1s=10")
    ap.add_argument("--target", type=float)
    args = ap.parse_args()
    if args.target is not None and (not math.isfinite(args.target) or args.target <= 0):
        ap.error("--target must be finite and positive")
    job, judged, shapes = load_run(args.run)
    base = score(shapes)
    print(f"job {job}: judged {judged:.3f}x, model {base:.3f}x")
    print(f"{'shape':6}{'ours':>8}{'sglang':>8}{'floor':>8}{'ours/floor':>11}{'+10us':>9}")
    for name, s in shapes.items():
        trial = {n: x["ours"] for n, x in shapes.items()}
        trial[name] -= 10
        print(f"{name:6}{s['ours']:8.1f}{s['sglang']:8.1f}{s['floor']:8.1f}"
              f"{s['ours'] / s['floor']:11.2f}{score(shapes, trial) - base:+9.4f}")
    if args.save:
        trial = {n: x["ours"] for n, x in shapes.items()}
        for item in args.save:
            name, us = item.split("=")
            trial[name] -= float(us)
            if not math.isfinite(trial[name]) or trial[name] <= 0:
                ap.error(f"savings must leave a finite positive latency: {item}")
        print(f"with savings {' '.join(args.save)}: {score(shapes, trial):.3f}x")
    if args.target:
        keep = base / args.target
        print(f"target {args.target:.3f}x: keep {keep:.3f} of every shape's time "
              f"(cut {100 * (1 - keep):.1f}%)")
    print(f"all shapes at TASK floors (idealized, not achievable forecast): "
          f"{score(shapes, {n: s['floor'] for n, s in shapes.items()}):.3f}x")


if __name__ == "__main__":
    main()
