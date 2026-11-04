#!/usr/bin/env python3
"""Bitwise comparison of two layer files on the same fresh inputs (box).

python tools/ly_bitcmp.py muse_decode_layer/cand/A.py muse_decode_layer/cand/B.py --judge J.json --router-seed S
    [--loads "decode 16,decode 32"] [--inputs 20] [--lock /root/okc/.build/box/gpu.lock] [--ls-cache F]
    [--pin-pdl8 1,8,16,32]

Both files are prepared on the same weights in one process, then every fresh input (f931 case seeds, local_start per
input) runs eagerly through both kernel_fn's; each output tensor is compared with torch.equal, and the reference
check runs on both outputs. Prints per cell: inputs compared, tensors that differ, the largest absolute difference,
and the two check verdicts.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--loads", default="decode 16,decode 32")
    ap.add_argument("--inputs", type=int, default=20)
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    ap.add_argument("--ls-cache", default="")
    ap.add_argument("--pin-pdl8", default="1,8,16,32")
    a = ap.parse_args()
    if a.lock:
        br.release_lock_for_triton()
        br.BudgetClock.install()
        br.LOCK = br.GpuLock(a.lock)
        br.LOCK.acquire("start")
    br.content_addressed_extensions(a.ext_cache)
    pdir = os.path.abspath("muse_decode_layer")
    sys.path.insert(0, pdir)
    ref = br.load_module("reference", os.path.join(pdir, "reference.py"))
    torch.backends.cuda.matmul.allow_tf32 = False
    _, cases = br.read_judge(a.judge)
    keep = {x.strip() for x in a.loads.split(",")}
    cases = [c for c in cases if c.get("load") in keep]
    weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=a.router_seed)
    pins = {int(t) for t in a.pin_pdl8.split(",") if t.strip()}
    impl = []
    for side, path in (("a", a.a), ("b", a.b)):
        mod = br.load_module(f"okc_{side}", path)
        if hasattr(mod, "_pdl8_setup"):
            mod._pdl8_setup = lambda st: None
        state = mod.prepare(weights)
        if br.pdl8_state(state) is not None:
            state["pdl8_on"] = {T: T in pins for T in getattr(mod, "PDL8_T", (1, 8, 16, 32))}
            state["pdl8_edges"] = {8: frozenset(mod.PDL8D_EDGES)} if 8 in pins else {}
        torch.cuda.synchronize()
        impl.append((mod.kernel_fn, state))
        print(f"BITCMP {side} {os.path.basename(path)} prepared", flush=True)
    lsc = br.LocalStartCache(a.ls_cache, os.path.join(pdir, "reference.py"), a.router_seed, ref.WEIGHTS_SEED) \
        if a.ls_cache else None
    allsame = True
    for case in cases:
        label = f"{case['load']} {case['layer']}"
        seeds = [case["seed"] + j for j in range(a.inputs)]
        ndiff, worst, checks = 0, 0.0, [0, 0]
        diffkeys = set()
        for s in seeds:
            extra = lsc.get(ref, case, s, weights) if lsc else ref.case_fields({**case, "seed": s}, weights)
            data = ref.generate_input(**{**case, "seed": s, **extra})
            outs = []
            for fn, state in impl:
                d = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
                out = br.call(fn, d, state)
                torch.cuda.synchronize()
                outs.append((d, out))
            raws = [ref.collect(d, out) for d, out in outs]   # hidden, stream, topk_ids, then the cache rows written
            if not torch.equal(raws[0], raws[1]):
                ndiff += 1
                pa, pb = ref.unpack(raws[0], data["config"]), ref.unpack(raws[1], data["config"])
                for i, (x, y) in enumerate(zip(pa, pb)):
                    if not torch.equal(x, y):
                        diffkeys.add(i)
                        if x.is_floating_point():
                            worst = max(worst, float((x.float() - y.float()).abs().max()))
            for k, raw in enumerate(raws):
                ok, _ = ref.check_implementation(outs[k][0], raw, weights)
                checks[k] += bool(ok)
        if lsc:
            lsc.save()
        allsame = allsame and ndiff == 0
        print(f"BITCMP {label}: inputs {len(seeds)} differing {ndiff} tensors {sorted(diffkeys)} max|diff| {worst:.3e}"
              f" check a {checks[0]}/{len(seeds)} b {checks[1]}/{len(seeds)}", flush=True)
    print("BITCMP ALL " + ("BITWISE EQUAL" if allsame else "DIFFER"), flush=True)


if __name__ == "__main__":
    main()
