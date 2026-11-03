"""Map the layer's per-kernel time against each input's MoE routing shape on the box.

python tools/box_route_map.py <layer submission.py> --router-seed S --judge J.json [--loads "decode 1,verify 1"]
    [--runs 60] [--mode busiest|case|rank:R] [--out map.json]

The reference's router_load / router_distinct targets (weights_fields, _mean_busiest_load) are
measured on each input's own busiest 32-expert range. --mode busiest gives every fresh input that
range as local_start; --mode case reuses the case seed's local_start for all inputs (what
box_run.py does); --mode rank:R takes the R-th busiest range (0 = busiest) to widen the sweep.
Per input it records routes, distinct experts and the most tokens on one expert in that range,
the cold graph time (net of the judge's fixed replay cost) and, from a second pass under
torch.profiler, every kernel's GPU time.
"""
import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import box_run  # noqa: E402


def shape_of(ref, data, weights, mode):
    ids = ref.ref_front(data, weights)["biased"].topk(ref.TOP_K, dim=-1).indices
    ranges = ref.N_EXPERTS // ref.LOCAL_EXPERTS
    pairs = torch.bincount((ids // ref.LOCAL_EXPERTS).flatten(), minlength=ranges)
    if mode == "case":
        ls = int(data["local_start"])
    else:
        r = 0 if mode == "busiest" else int(mode.split(":")[1])
        ls = ref.LOCAL_EXPERTS * int(pairs.argsort(descending=True, stable=True)[r])
    local = ids[(ids - ls) % ref.N_EXPERTS < ref.LOCAL_EXPERTS]
    per = torch.bincount(local.flatten() - ls, minlength=ref.LOCAL_EXPERTS) if local.numel() else torch.zeros(1)
    return ls, dict(routes=int(local.numel()), distinct=int((per > 0).sum()), max_tok=int(per.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--loads", default="decode 1,verify 1")
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--mode", default="busiest")
    ap.add_argument("--seed-offset", type=int, default=0, help="shift fresh input seeds off the judge's")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    pdir = os.path.abspath("muse_decode_layer")
    sys.path.insert(0, pdir)
    ref = box_run.load_module("reference", os.path.join(pdir, "reference.py"))
    torch.backends.cuda.matmul.allow_tf32 = False
    _, judge_cases = box_run.read_judge(a.judge)
    cases = judge_cases or [dict(c, seed=1000 + i) for i, c in enumerate(box_run.CASES["muse_decode_layer"])]
    keep = {s.strip() for s in a.loads.split(",")}
    cases = [c for c in cases if c.get("load") in keep]
    weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=a.router_seed)
    mod = box_run.load_module("okc_submission", a.submission)
    state = mod.prepare(weights)
    torch.cuda.synchronize()
    fn = mod.kernel_fn
    timer = box_run.Timer()
    out = []
    for case in cases:
        case = dict(case)
        case.update(ref.case_fields(case, weights))
        seeds = [case["seed"] + a.seed_offset + j for j in range(a.runs)]
        fresh, shapes = [], []
        for s in seeds:
            d = ref.generate_input(**{**case, "seed": s})
            ls, sh = shape_of(ref, d, weights, a.mode)
            d["local_start"].fill_(ls)
            fresh.append(d)
            shapes.append(dict(sh, local_start=ls))
        static = ref.generate_input(**{**case, "seed": seeds[0]})
        g, _ = box_run.capture(fn, static, state)
        ov = timer.overhead()["overhead_ns"]
        net = []
        for d in fresh:
            box_run.copy_into(static, d)
            torch.cuda.synchronize()
            net.append((timer.cold(g) - ov) / 1e3)
        from torch.profiler import ProfilerActivity, profile
        kern = []
        for d in fresh:
            box_run.copy_into(static, d)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                timer.cold(g)
            agg = {}
            for ev in prof.events():
                if (ev.device_type == torch.autograd.DeviceType.CUDA and "reduce_kernel" not in ev.name
                        and not ev.name.startswith(("Memcpy", "Memset"))):
                    agg[ev.name] = agg.get(ev.name, 0.0) + ev.device_time
            kern.append(agg)
        label = f"{case['load']} {case['layer']}"
        rows = [dict(sh, net_us=n, moe_us=sum(v for k, v in kv.items() if "moe_fused" in k), kernels=kv)
                for sh, n, kv in zip(shapes, net, kern)]
        out.append(dict(case=case, rows=rows))
        print(f"MAP {label} mode {a.mode}: net med {statistics.median(net):.2f}, moe_fused med "
              f"{statistics.median(r['moe_us'] for r in rows):.2f}, mean routes "
              f"{statistics.mean(r['routes'] for r in rows):.2f}, mean distinct "
              f"{statistics.mean(r['distinct'] for r in rows):.2f}", flush=True)
        for dist in sorted({r["distinct"] for r in rows}):
            b = [r for r in rows if r["distinct"] == dist]
            print(f"MAP {label} distinct {dist:2d}: n {len(b):2d} routes {statistics.mean(r['routes'] for r in b):5.1f}"
                  f" max_tok {statistics.mean(r['max_tok'] for r in b):4.1f} net {statistics.median(r['net_us'] for r in b):7.2f}"
                  f" moe_fused {statistics.median(r['moe_us'] for r in b):6.2f}"
                  f" (min {min(r['moe_us'] for r in b):6.2f} max {max(r['moe_us'] for r in b):6.2f})", flush=True)
        del fresh, g
        torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
