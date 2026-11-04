"""CL-0.5: bitwise check of cl_math.cuh against the helper0 kernel of the public MoE file.

Reference: the public file's helper0 source compiled with its own flags, plus a debug copy
(helper0_debug/build_debug.py) that also dumps each phase B unit's partial sum of squares and the routing tables.
The plain build and the debug build must agree bitwise on inter and ybuf (the dumps do not change the math).
Candidate: test_unit_math.cu, one CTA per unit, using only cl_math.cuh.

Per T in --tokens, per input: M = T tokens; three local experts take every token (units with exactly T tokens),
one local expert takes a random subset, the other slots are random experts of 256 (some local). Each routed local
expert's 512 phase A and 256 phase B units are computed by the candidate over all its tokens (ceil(n / 8) n-tiles)
and compared bitwise with helper0's inter rows, ybuf rows and per-unit sums of squares; the final pass (route
scale and slot-order weighted add) is recomputed from helper0's ss and ybuf and compared with its output.

T = 1 runs the TINY build, T = 8 TINY2, T = 32 the MID defines (helper0 with OKC_SPLIT=1: experts of more than
8 tokens become jobs of at most 8). The public file runs B4 (the wgmma source) at T = 32; this check covers the
helper0 math only.

Usage (box): python muse_moe_w4a16/cl/tests/run_unit_math.py --lock /root/okc/.build/box/gpu.lock
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MOE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, MOE)
sys.path.insert(0, os.path.join(HERE, "helper0_debug"))

import build_debug  # noqa: E402

PUBLIC = os.path.join(MOE, "cand", "current_control_release.py")
EMAX = 384 * 8


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_public():
    spec = importlib.util.spec_from_file_location("cl05_public", PUBLIC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_defines(pub):
    return {1: pub.MOE_TINY_DEFINES, 8: pub.MOE_TINY2_DEFINES, 32: pub.MOE_MID_DEFINES}


def build_test_ext(pub, negctl=False):
    from torch.utils.cpp_extension import load
    flags = [f for f in build_debug.cflags(pub, ()) if not f.startswith("-DOKC_")] + (["-DCL05_NEGCTL"] if negctl else [])
    tag = hashlib.md5(open(os.path.join(HERE, "test_unit_math.cu"), "rb").read()
                      + open(os.path.join(os.path.dirname(HERE), "cl_math.cuh"), "rb").read()).hexdigest()[:12]
    return load(name=f"cl05_unit_math_{tag}" + ("_neg" if negctl else ""), sources=[os.path.join(HERE, "test_unit_math.cu")],
                extra_cuda_cflags=flags, extra_include_paths=[os.path.dirname(HERE)], verbose=False)


@contextlib.contextmanager
def gpu_lock(path):
    if not path:
        yield
        return
    import fcntl
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    t0 = time.time()
    fcntl.flock(fd, fcntl.LOCK_EX)
    print(f"LOCK acquired after {time.time() - t0:.1f} s", flush=True)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def make_weights(seed, dev):
    from nvfp4 import random_weights
    gen = torch.Generator(device=dev).manual_seed(seed)
    w13, w13_sf, gs13 = random_weights(8192, 4096, generator=gen, device=dev, leading=(32,))
    w2, w2_sf, gs2 = random_weights(4096, 4096, generator=gen, device=dev, leading=(32,))
    # per-expert global scales (the reference uses one value for all experts) so a wrong expert index shows
    f13 = torch.exp2(torch.rand((32,), generator=gen, device=dev) * 2 - 1)
    f2 = torch.exp2(torch.rand((32,), generator=gen, device=dev) * 2 - 1)
    w = {"w13": w13, "w13_sf": w13_sf, "w13_gs": (gs13 * f13).float().contiguous(),
         "w2": w2, "w2_sf": w2_sf, "w2_gs": (gs2 * f2).float().contiguous(),
         "post_norm_w": (0.1 * torch.randn((4096,), generator=gen, device=dev)).to(torch.bfloat16)}
    return w


def make_input(T, seed, dev, outliers):
    from nvfp4 import outlier_channels
    g = torch.Generator(device="cpu").manual_seed(seed)
    full = torch.randperm(32, generator=g)[:4].tolist()
    if T == 1:
        ids = [torch.randperm(32, generator=g)[:8].tolist()]
    else:
        f, part = full[:3], full[3]
        subset = set(torch.randperm(T, generator=g)[: int(torch.randint(1, T, (1,), generator=g))].tolist())
        ids = []
        for t in range(T):
            row = list(f) + ([part] if t in subset else [])
            while len(row) < 8:
                e = int(torch.randint(0, 256, (1,), generator=g))
                if e not in row and e != part:
                    row.append(e)
            perm = torch.randperm(8, generator=g).tolist()
            ids.append([row[i] for i in perm])
    ids = torch.tensor(ids, dtype=torch.int32, device=dev)
    tw = torch.rand((T, 8), generator=g) + 0.05
    tw = (tw / tw.sum(-1, keepdim=True)).float().to(dev)
    gd = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((T, 4096), generator=gd, device=dev).to(torch.bfloat16)
    if outliers:
        x = outlier_channels(x, generator=gd)
    cfg = {"bs": T, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0, "local_count": 32,
           "top_k": 8, "sharding": "ep8"}
    return {"hidden_states": x.contiguous(), "topk_ids": ids.contiguous(), "topk_weights": tw.contiguous(),
            "config": cfg}


def run_helper0(pub, ext, st, data, dbg):
    scratch = pub._ma_make_scratch(st["dev"])
    if dbg:
        scratch = scratch + [torch.zeros(build_debug.DBG_WORDS, dtype=torch.int32, device=st["dev"])]
    out = torch.empty_like(data["hidden_states"])
    ext.okc_run(st["ma_w"], scratch, data["hidden_states"], data["topk_ids"], data["topk_weights"], out, 0, 32,
                st["nsm"])
    torch.cuda.synchronize()
    return out, scratch


def bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def check_input(pub, test, st, exts, T, data, reference, wref):
    out_p, scr_p = run_helper0(pub, exts["pub"], st, data, False)
    out_d, scr_d = run_helper0(pub, exts["dbg"], st, data, True)
    inter_p, inter_d = scr_p[2][: EMAX * 4096].view(EMAX, 4096), scr_d[2][: EMAX * 4096].view(EMAX, 4096)
    ybuf_p = scr_p[1][EMAX: EMAX + EMAX * 4096].view(EMAX, 4096)
    ybuf_d = scr_d[1][EMAX: EMAX + EMAX * 4096].view(EMAX, 4096)
    ss_d = scr_d[1][:EMAX]
    dbgw = scr_d[3]
    sq_d = dbgw[: build_debug.DBG_SQ_FLOATS].view(torch.float32).view(EMAX, 256)
    accA_d = dbgw[build_debug.DBG_ACC_A_OFF: build_debug.DBG_ACC_B_OFF].view(torch.float32).view(EMAX, 8192)
    accB_d = dbgw[build_debug.DBG_ACC_B_OFF: build_debug.DBG_WORDS].view(torch.float32).view(EMAX, 4096)
    rt = dbgw[build_debug.DBG_ROUTE_OFF:].cpu()
    njF, njS = int(rt[0]), int(rt[1])
    assert njF == 0, njF
    jobs = [(int(rt[2 + 3 * j]), int(rt[3 + 3 * j]), int(rt[4 + 3 * j])) for j in range(njS)]
    pos = dbgw[build_debug.DBG_POS_OFF: build_debug.DBG_POS_OFF + T * 8].contiguous()
    pos_c = pos.cpu().tolist()
    nE = sum(c for _, _, c in jobs)
    r = {}
    r["plain_vs_debug_inter_eq"] = bool(torch.equal(bits(inter_p[:nE]), bits(inter_d[:nE])))
    r["plain_vs_debug_ybuf_eq"] = bool(torch.equal(bits(ybuf_p[:nE]), bits(ybuf_d[:nE])))
    r["plain_vs_debug_out_diff"] = int((bits(out_p) != bits(out_d)).sum())
    # experts: (e, first position, count) from the jobs (a split expert has consecutive jobs)
    experts = {}
    for e, s0, c in jobs:
        if e in experts:
            p0, n = experts[e]
            assert p0 + n == s0
            experts[e] = (p0, n + c)
        else:
            experts[e] = (s0, c)
    ent_tok = [None] * nE
    for i, q in enumerate(pos_c):
        if q >= 0:
            ent_tok[q] = i >> 3
    ids = data["topk_ids"].cpu()
    for e, (p0, n) in experts.items():  # the routing dump agrees with topk_ids
        toks = sorted(int(t) for t in (ids == e).nonzero()[:, 0].tolist())
        assert toks == ent_tok[p0: p0 + n], (e, toks, ent_tok[p0: p0 + n])
    dev = st["dev"]
    w = st["ma_w"]
    cnt = {"A_units": 0, "A_bad": 0, "B_units": 0, "B_bad_y": 0, "B_bad_sq": 0, "A_units_T": 0, "B_units_T": 0,
           "A_bad_acc": 0, "B_bad_acc": 0}
    by_ntt = {}
    for e, (p0, n) in experts.items():
        by_ntt.setdefault((n + 7) // 8, []).append((e, p0, n))
    for ntt, group in sorted(by_ntt.items()):
        rowsA, rowsB, uA, uB = [], [], [], []
        for e, p0, n in group:
            ra0, rb0 = len(rowsA), len(rowsB)
            rowsA += ent_tok[p0: p0 + n]
            rowsB += list(range(p0, p0 + n))
            uA += [(e, rb, ra0, n, p0) for rb in range(512)]
            uB += [(e, rb, rb0, n, p0) for rb in range(256)]
        def cols(us, k):
            return torch.tensor([u[k] for u in us], dtype=torch.int32, device=dev)
        ta = torch.tensor(rowsA, dtype=torch.int32, device=dev)
        tb = torch.tensor(rowsB, dtype=torch.int32, device=dev)
        outA, accA = test.unit_a(w[10], w[2], data["hidden_states"], cols(uA, 0), cols(uA, 1), cols(uA, 2), cols(uA, 3), ta, ntt)
        outY, outS, accB = test.unit_b(w[11], w[5], w[7], inter_d, cols(uB, 0), cols(uB, 1), cols(uB, 2), cols(uB, 3), tb, ntt)
        torch.cuda.synchronize()
        # reference gathers: unit (e, rb), token i -> inter[p0 + i, 8 rb + g], ybuf[p0 + i, 16 rb + g], sq[p0 + i, rb]
        for us, phase in ((uA, "A"), (uB, "B")):
            ue = cols(us, 0).long(); urb = cols(us, 1).long(); un = cols(us, 3).long(); up0 = cols(us, 4).long()
            i = torch.arange(32, device=dev)
            valid = i[None, :] < un[:, None]
            prow = (up0[:, None] + i[None, :]).clamp(max=EMAX - 1)
            acc_ref = (accA_d if phase == "A" else accB_d)[prow[:, :, None], (urb[:, None, None] * 16 + torch.arange(16, device=dev))]
            acc_out = accA if phase == "A" else accB
            cnt[phase + "_bad_acc"] += int(((bits(acc_out) != bits(acc_ref)) & valid[:, :, None]).any(-1).any(-1).sum())
            if phase == "A":
                ref = inter_d[prow[:, :, None], (urb[:, None, None] * 8 + torch.arange(8, device=dev))]
                bad = ((bits(outA) != bits(ref)) & valid[:, :, None]).any(-1).any(-1)
                cnt["A_units"] += len(us); cnt["A_bad"] += int(bad.sum())
                cnt["A_units_T"] += int((un == T).sum())
            else:
                ref = ybuf_d[prow[:, :, None], (urb[:, None, None] * 16 + torch.arange(16, device=dev))]
                bad = ((bits(outY) != bits(ref)) & valid[:, :, None]).any(-1).any(-1)
                refs = sq_d[prow, urb[:, None].expand(-1, 32)]
                bads = ((bits(outS) != bits(refs)) & valid).any(-1)
                cnt["B_units"] += len(us); cnt["B_bad_y"] += int(bad.sum()); cnt["B_bad_sq"] += int(bads.sum())
                cnt["B_units_T"] += int((un == T).sum())
    r.update(cnt)
    # the row totals of ss are atomic sums in arrival order; the per-unit partials summed in rb order agree to
    # within that reordering only
    tot = torch.zeros(nE, dtype=torch.float32, device=dev)
    for e, (p0, n) in experts.items():
        tot[p0: p0 + n] = sq_d[p0: p0 + n].sum(-1)
    r["ss_reorder_max_rel"] = float(((tot - ss_d[:nE]).abs() / ss_d[:nE].abs()).max())
    out_c = test.combine(pos, data["topk_weights"].view(-1).contiguous(), ss_d.contiguous(), ybuf_d.contiguous(), T)
    torch.cuda.synchronize()
    r["final_diff_elems"] = int((bits(out_c) != bits(out_d)).sum())
    ok, msg = reference.check_implementation(data, out_d, wref)
    r["reference_check"] = bool(ok)
    r["reference_msg"] = str(msg)[:160]
    r["experts"] = {int(e): n for e, (_, n) in experts.items()}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,8,32")
    ap.add_argument("--inputs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=505)
    ap.add_argument("--lock", default="")
    ap.add_argument("--build-only", default="", help="pub:T | dbg:T | test: build one extension and exit")
    ap.add_argument("--json", default="")
    ap.add_argument("--negctl", action="store_true", help="reverse the cross-warp sum in the test kernel (must fail)")
    a = ap.parse_args()
    pub = load_public()
    sfx = pub._MA_SFX_IF_PRMT
    defs = build_defines(pub)
    Ts = [int(t) for t in a.tokens.split(",")]
    if a.build_only:
        kind, _, t = a.build_only.partition(":")
        if kind == "test":
            build_test_ext(pub, a.negctl)
        elif kind == "pub":
            pub._ma_ext(tuple(defs[int(t)]) + tuple(sfx))
        else:
            build_debug.build(pub, defs[int(t)], sfx)
        print("BUILT", a.build_only)
        return
    path, dsha = build_debug.write_source(pub)
    print(f"public {sha(PUBLIC)[:16]} cl_math {sha(os.path.join(os.path.dirname(HERE), 'cl_math.cuh'))[:16]} "
          f"test {sha(os.path.join(HERE, 'test_unit_math.cu'))[:16]} debug_src {dsha[:16]}", flush=True)
    t0 = time.time()
    exts = {T: {"pub": pub._ma_ext(tuple(defs[T]) + tuple(sfx)), "dbg": build_debug.build(pub, defs[T], sfx)}
            for T in Ts}
    test = build_test_ext(pub, a.negctl)
    print(f"builds {time.time() - t0:.1f} s", flush=True)
    import reference
    results = {}
    with gpu_lock(a.lock):
        dev = torch.device("cuda")
        wref = make_weights(a.seed, dev)
        local = {k: wref[k] for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
        local["post_expert_norm"] = wref["post_norm_w"]
        assert pub._ma_scales_prmt_exact(local["w13_sf"], local["w2_sf"]), "PRMT scale path not exact for these weights"
        st = {"ma_sc6": False, "guards": [], "dev": dev,
              "nsm": torch.cuda.get_device_properties(dev).multi_processor_count}
        pub._fork_native_ma_prepare(local, st)
        torch.cuda.synchronize()
        for T in Ts:
            for k in range(a.inputs):
                data = make_input(T, a.seed * 1000 + T * 10 + k, dev, outliers=(k == a.inputs - 1))
                r = check_input(pub, test, st, exts[T], T, data, reference, wref)
                results[f"T{T}_in{k}"] = r
                print(f"T={T} input {k}: " + json.dumps(r), flush=True)
    keys = ("A_bad_acc", "B_bad_acc", "A_units", "A_bad", "A_units_T", "B_units", "B_bad_y", "B_bad_sq", "B_units_T", "final_diff_elems")
    for T in Ts:
        rs = [v for k, v in results.items() if k.startswith(f"T{T}_")]
        s = {k: sum(r[k] for r in rs) for k in keys}
        s["plain_vs_debug_all_eq"] = all(r["plain_vs_debug_inter_eq"] and r["plain_vs_debug_ybuf_eq"] for r in rs)
        s["reference_all_pass"] = all(r["reference_check"] for r in rs)
        s["ss_reorder_max_rel"] = max(r["ss_reorder_max_rel"] for r in rs)
        print(f"SUMMARY T={T} " + json.dumps(s), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
