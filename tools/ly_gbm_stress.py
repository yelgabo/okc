"""moe-gbm stress for MFIN on the MAIN (RFIN) MoE build, run on the box under gpu.lock.

For each M (all served by the MAIN build: M > 32) it captures one CUDA graph of the MoE launch per file (control and
candidate) on static buffers, then for every replay writes a new routing (uniform, all-local, hot-expert, sparse
mixes) into those buffers, replays both graphs and checks:
  - the candidate output equals the control output bitwise;
  - the candidate's MFIN counters for this launch's parity equal the target 64 * njobs(e) of every local expert
    exactly (so releases == target: one per warpgroup per phase B unit), and the other parity is all zero (reset);
  - the parity alternates between consecutive launches.
Usage: python tools/ly_gbm_stress.py CTL.py CAND.py --router-seed S --ms 33,128,256 --reps 2000,2000,2000
"""
import argparse
import importlib.util
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def routing(M, gen, dev, kind):
    if kind == "uniform":
        ids = torch.rand((M, 256), generator=gen, device=dev).argsort(1)[:, :8]
    elif kind == "local":
        ids = torch.rand((M, 32), generator=gen, device=dev).argsort(1)[:, :8]
    elif kind == "hot":
        k = int(torch.randint(8, 13, (1,), generator=gen, device=dev))
        base = torch.rand((32,), generator=gen, device=dev).argsort()[:k]
        ids = base[torch.rand((M, k), generator=gen, device=dev).argsort(1)[:, :8]]
    else:  # mixed: each row a random number of local slots with skewed expert weights, the rest remote or -1
        w = torch.rand((32,), generator=gen, device=dev).pow(4) + 1e-3
        loc = torch.multinomial(w.expand(M, 32), 8, replacement=False, generator=gen)
        rem = 32 + torch.rand((M, 224), generator=gen, device=dev).argsort(1)[:, :8]
        nl = torch.randint(0, 9, (M, 1), generator=gen, device=dev)
        col = torch.arange(8, device=dev)[None, :]
        ids = torch.where(col < nl, loc, rem)
        drop = torch.rand((M, 8), generator=gen, device=dev) < 0.05
        ids = torch.where(drop, torch.full_like(ids, -1), ids)
    return ids.to(torch.int32).contiguous()


def target(ids, sk_t=8, ft_t=64):
    v = ids.flatten().long()
    tot = torch.bincount(v[(v >= 0) & (v < 32)], minlength=32)
    nj = torch.where(tot > sk_t, (tot + ft_t - 1) // ft_t, (tot > 0).long())
    return 64 * nj, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ctl")
    ap.add_argument("cand")
    ap.add_argument("--router-seed", type=int, default=None)
    ap.add_argument("--ms", default="33,128,256")
    ap.add_argument("--reps", default="2000,2000,2000")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    a = ap.parse_args()
    import bx_box_run
    bx_box_run.content_addressed_extensions(a.ext_cache)
    ref = load("reference", "muse_decode_layer/reference.py")
    weights = (ref.generate_weights(ref.WEIGHTS_SEED, router_seed=a.router_seed) if a.router_seed is not None
               else ref.generate_weights(ref.WEIGHTS_SEED))
    mods, states = [], []
    for name, path in (("okc_ctl", a.ctl), ("okc_cand", a.cand)):
        t0 = time.time()
        m = load(name, path)
        m._pdl8_setup = lambda st: None
        states.append(m.prepare(weights))
        mods.append(m)
        print(f"STRESS prepare {path} {time.time() - t0:.0f} s, moe status {states[-1].get('_fork_moe_status')}", flush=True)
    dev = weights["w13"].device
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    emax = mods[1]._MA_EMAX
    mfl0 = 8 + 1024 + emax
    gen = torch.Generator(device=dev).manual_seed(20261104)
    kinds = ("uniform", "local", "hot", "mixed")
    all_ok = True
    for M, R in zip([int(x) for x in a.ms.split(",")], [int(x) for x in a.reps.split(",")]):
        x = torch.empty((M, 4096), device=dev, dtype=torch.bfloat16)
        ids = torch.empty((M, 8), device=dev, dtype=torch.int32)
        tw = torch.empty((M, 8), device=dev, dtype=torch.float32)
        outs, graphs = [], []
        x.copy_(torch.randn((M, 4096), generator=gen, device=dev).to(torch.bfloat16))
        ids.copy_(routing(M, gen, dev, "local"))
        tw.copy_(torch.rand((M, 8), generator=gen, device=dev) / 4)
        for m, st in zip(mods, states):
            ext = m._fork_native_moe(M, st)
            out = torch.empty_like(x)
            run = lambda ext=ext, st=st, out=out: ext.okc_run(st["ma_w"], st["ma_scratch"], x, ids, tw, out, 0, 32, nsm)
            run()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()
            outs.append(out)
            graphs.append(g)
        keys = [k for k, v in states[1]["ma_ext"].items() if v is mods[1]._fork_native_moe(M, states[1])]
        sc = states[1]["ma_scratch"][0]
        bad_out = bad_cnt = bad_zero = bad_par = 0
        last_gen = None
        hist = {"fat_jobs": 0, "skinny_jobs": 0, "max_jobs_per_expert": 0, "max_tot": 0}
        t0 = time.time()
        for r in range(R):
            kind = kinds[r % len(kinds)]
            x.copy_(torch.randn((M, 4096), generator=gen, device=dev).to(torch.bfloat16))
            ids.copy_(routing(M, gen, dev, kind))
            tw.copy_(torch.rand((M, 8), generator=gen, device=dev) / 4)
            graphs[0].replay()
            graphs[1].replay()
            tg, tot = target(ids)
            ce = sc[mfl0:mfl0 + 64].view(2, 32).long()
            eq = (ce == tg[None, :]).all(1)
            zero = (ce == 0).all(1)
            same = torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
            eqc, zc = eq.tolist(), zero.tolist()
            g = 0 if (eqc[0] and zc[1]) else 1 if (eqc[1] and zc[0]) else -1
            if not same:
                bad_out += 1
            if g < 0:
                if not (eqc[0] or eqc[1]):
                    bad_cnt += 1
                else:
                    bad_zero += 1
                if bad_cnt + bad_zero <= 3:
                    print(f"STRESS M={M} r={r} {kind} counters {ce.tolist()} target {tg.tolist()}", flush=True)
            elif last_gen is not None and g == last_gen and int(tg.sum()) > 0:
                bad_par += 1
            last_gen = g if (g >= 0 and int(tg.sum()) > 0) else None
            fat = (tot > 8)
            hist["fat_jobs"] += int(((tot + 63) // 64)[fat].sum())
            hist["skinny_jobs"] += int(((tot > 0) & ~fat).sum())
            hist["max_jobs_per_expert"] = max(hist["max_jobs_per_expert"], int((tg // 64).max()))
            hist["max_tot"] = max(hist["max_tot"], int(tot.max()))
        torch.cuda.synchronize()
        ok = bad_out == bad_cnt == bad_zero == bad_par == 0
        all_ok &= ok
        print(f"STRESS M={M} build {keys} replays {R} ({time.time() - t0:.0f} s): output mismatches {bad_out}, "
              f"counter != target {bad_cnt}, other parity nonzero {bad_zero}, parity repeats {bad_par}; {hist}; "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        del graphs, outs
    print(f"STRESS {'PASS' if all_ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
