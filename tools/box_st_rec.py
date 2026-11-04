"""Per-CTA weight-copy record of the layer's fused MoE (helper0 TINY2 at verify 1, MID at decode 16).

python tools/box_st_rec.py muse_decode_layer/cand/ly_st_rec.py --judge J.json --router-seed S [--inputs 10]
    [--loads "verify 1,decode 16"] [--lock F] [--ls-cache F] [--out f.json]

The file comes from tools/dl_ly_st_patch.py --variant rec. For each fresh input the layer runs eagerly once, the
MoE launch's arguments are recorded and replayed once after a 1 GiB L2 flush with the record tensor: every weight
copy a CTA issued (globaltimer, phase, expert block), the time warp 0 saw each ring slot full, and the MOE-TL-01
stamps (unit retires). Prints the concurrent address pattern across CTAs and where warp 0 waited for weights.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402
import box_moe_tl8 as tl8  # noqa: E402

import torch  # noqa: E402

SLOTS, NC, RN = 64, 136, 128
CNT = NC * SLOTS
ISS = CNT + NC
RDY = ISS + NC * RN * 2
WT = RDY + NC * RN
TOTAL = WT + NC * RN
UNIT = 36864
RBA, RBB = 512, 256


class Recorder:
    def __init__(self, ext, rows):
        self.ext, self.rows, self.calls = ext, rows, []

    def okc_run(self, *args):
        if len(args[1]) == 3 and args[2].shape[0] == self.rows:
            self.calls.append([x.clone() if torch.is_tensor(x) else x for x in args])
        return self.ext.okc_run(*args)

    def __getattr__(self, name):
        return getattr(self.ext, name)


def decode(buf, nsm):
    b = buf.cpu().tolist()
    t0 = min(b[c * SLOTS] for c in range(nsm) if b[c * SLOTS] > 0)
    ctas = []
    for c in range(nsm):
        cnt = b[c * SLOTS + 17]
        n = min((cnt & 0xFFFF) + ((cnt >> 16) & 0xFFFF), RN)
        iss = []
        for k in range(n):
            t, w = b[ISS + (c * RN + k) * 2], b[ISS + (c * RN + k) * 2 + 1]
            pb, blk = (w >> 31) & 1, w & 0x7FFFFFFF
            rbs = RBB if pb else RBA
            iss.append(dict(t=(t - t0) / 1e3, pb=pb, e=blk // rbs, rb=blk % rbs, blk=blk))
        rdy = [(b[RDY + c * RN + q] - t0) / 1e3 if b[RDY + c * RN + q] > 0 else None for q in range(RN)]
        wt = [b[WT + c * RN + q] / 1e3 for q in range(RN)]
        ret = [(b[c * SLOTS + 18 + q] - t0) / 1e3 if b[c * SLOTS + 18 + q] > 0 else None for q in range(46)]
        ctas.append(dict(smid=b[CNT + c], abuf=b[c * SLOTS + 14] / 1e3, iss=iss, rdy=rdy, wt=wt, ret=ret, nA=cnt & 0xFFFF, nB=(cnt >> 16) & 0xFFFF,
                         entry=(b[c * SLOTS] - t0) / 1e3))
    return ctas


def analyse(ctas, label):
    nsm = len(ctas)
    # address of each copy: phase A units at e * 512 + rb, phase B after all 32 x 512 phase A units
    lat, stallA, stallB, stall_at = [], [], [], {}
    for c in ctas:
        for q, u in enumerate(c["iss"]):
            r = c["rdy"][q] if q < RN else None
            if r is not None:
                lat.append((u["pb"], r - u["t"]))
        for q in range(min(RN, len(c["iss"]))):
            if c["rdy"][q] is None:
                continue
            w = c["wt"][q]
            ph = "B" if q >= c["nA"] else "A"
            (stallB if ph == "B" else stallA).append(w)
            if q == c["nA"]:
                key = "AtoB"
            elif q < 3:
                key = ph + "_first3"
            elif c["iss"][q]["e"] != c["iss"][q - 1]["e"]:
                key = ph + "_jobswitch"
            elif q - c["nA"] in (1, 2):
                key = "B_first3"
            else:
                key = ph + "_steady"
            stall_at.setdefault(key, []).append(w)
    # concurrency: units in flight (issued, not yet seen full) at the middle of phase A
    endA = statistics.median(c["rdy"][c["nA"] - 1] for c in ctas if 0 < c["nA"] <= RN and c["rdy"][c["nA"] - 1])
    snaps = {}
    for frac in (0.3, 0.6):
        t = endA * frac
        fl = []
        for ci, c in enumerate(ctas):
            for q, u in enumerate(c["iss"]):
                r = c["rdy"][q] if q < RN else None
                if u["t"] <= t and (r is None or r > t):
                    fl.append((ci, u))
        exps = sorted({(u["pb"], u["e"]) for _, u in fl})
        regions = {((u["pb"] * 32 * RBA + u["e"] * (RBB if u["pb"] else RBA) + u["rb"]) * UNIT) >> 21 for _, u in fl}
        heads = {}
        for ci, u in fl:
            heads.setdefault(ci, u)
        hs = sorted(heads.items())
        strides = [hs[i + 1][1]["blk"] - hs[i][1]["blk"] for i in range(len(hs) - 1)
                   if hs[i + 1][1]["e"] == hs[i][1]["e"] and hs[i + 1][1]["pb"] == hs[i][1]["pb"]]
        snaps[frac] = dict(t=t, inflight=len(fl), ctas=len(heads), experts=len(exps), regions_2MB=len(regions),
                           stride_med=statistics.median(strides) if strides else None)
    q = lambda xs: (statistics.median(xs), sorted(xs)[int(0.9 * (len(xs) - 1))], sum(xs)) if xs else (0, 0, 0)  # noqa: E731
    latA = [x for pb, x in lat if not pb]
    latB = [x for pb, x in lat if pb]
    out = dict(label=label, endA=endA, a_buf=statistics.mean(c["abuf"] for c in ctas), snaps=snaps, latA=q(latA), latB=q(latB),
               stallA_per_cta=sum(stallA) / nsm, stallB_per_cta=sum(stallB) / nsm,
               stall_at={k: (len(v), sum(v) / nsm, statistics.median(v)) for k, v in stall_at.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--loads", default="verify 1,decode 16")
    ap.add_argument("--inputs", type=int, default=10)
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    ap.add_argument("--ls-cache", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--extra", default="", help="comma-separated defines for the recorded launches only, e.g. "
                    "-DST_NOMATH=1,-DST_SPIN=900 (the prepared state and its checks use the file's own builds)")
    ap.add_argument("--replay", default="", help="also stream each input's recorded phase A sequence from the layer's own "
                    "w13 tensor with muse_moe_w4a16/cl/st_replay.cu: comma-separated consumer delays in ns")
    ap.add_argument("--flush", default="write", choices=("write", "read"),
                    help="L2 flush before the recorded launch: write (fill_, box_moe_tl8.py's) or read (sum, the judge's)")
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
    mod = br.load_module("okc_submission", a.submission)
    if hasattr(mod, "_pdl8_setup"):
        mod._pdl8_setup = lambda st: None
    state = mod.prepare(weights)
    state["pdl8_on"] = {T: True for T in getattr(mod, "PDL8_T", (1, 8, 16, 32))}
    state["pdl8_edges"] = {8: frozenset(mod.PDL8D_EDGES)}
    torch.cuda.synchronize()
    mod._moe(8, state)
    mod._moe(16, state)
    recs = {"tiny2": Recorder(state["ma_ext"]["tiny2"], 8), "mid": Recorder(state["ma_ext"]["mid"], 16)}
    extra = tuple(x for x in a.extra.split(",") if x)
    run_ext = {k: (mod._ma_ext(tuple(d) + extra + tuple(state["ma_sfx"])) if extra else r.ext)
               for (k, r), d in zip(recs.items(), (mod.MOE_TINY2_DEFINES, mod.MOE_MID_DEFINES))}
    for k, r in recs.items():
        state["ma_ext"][k] = r
    lsc = br.LocalStartCache(a.ls_cache, os.path.join(pdir, "reference.py"), a.router_seed, ref.WEIGHTS_SEED) \
        if a.ls_cache else None
    flush = torch.ones(1 << 28, dtype=torch.float32, device="cuda")
    dump = []
    rp = None
    if a.replay:
        sys.path.insert(0, os.path.abspath("muse_moe_w4a16/cl"))
        import st_replay_bench as rpb
        rp = rpb.build()
    for case in cases:
        res = []
        for j in range(a.inputs):
            seed = case["seed"] + j
            ls = (lsc.get(ref, case, seed, weights) if lsc else ref.case_fields({**case, "seed": seed}, weights))
            data = ref.generate_input(**{**case, "seed": seed, "local_start": ls["local_start"]})
            for r in recs.values():
                r.calls.clear()
            with torch.no_grad():
                mod.kernel_fn(data, state)
            torch.cuda.synchronize()
            key = next((k for k, r in recs.items() if r.calls), None)
            rec = recs[key] if key else None
            if rec is None:
                print("REC no helper0 launch recorded", case["load"], flush=True)
                continue
            w_, scr, x, lids, wts, out, l0, l1, nsm = rec.calls[-1]
            plain_us = []
            for _ in range(3):
                if a.flush == "write":
                    flush.fill_(1.0)
                else:
                    flush.sum()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                run_ext[key].okc_run(w_, scr, x, lids, wts, torch.empty_like(out), l0, l1, nsm)
                e1.record()
                torch.cuda.synchronize()
                plain_us.append(e0.elapsed_time(e1) * 1e3)
            buf = torch.zeros(TOTAL, dtype=torch.int64, device="cuda")
            inst = torch.empty_like(out)
            if a.flush == "write":
                flush.fill_(1.0)
            else:
                flush.sum()
            torch.cuda.synchronize()
            run_ext[key].okc_run(w_, list(scr) + [buf], x, lids, wts, inst, l0, l1, nsm)
            torch.cuda.synchronize()
            if int(buf[: nsm * SLOTS].abs().sum()) == 0:
                print(f"REC plain-only {case['load']} {case['layer']} seed {seed} plain_us {statistics.median(plain_us):.2f}",
                      flush=True)
                continue
            ctas = decode(buf, nsm)
            st = tl8.stats(buf[: nsm * SLOTS].view(nsm, SLOTS).cpu(), lids)
            an = analyse(ctas, f"{case['load']} {case['layer']} seed {seed}")
            an.update(plain_us=statistics.median(plain_us), span=st["span"], A=st["A"], B=st["B"], a_wwait=st["a_wwait"], b_wwait=st["b_wwait"],
                      distinct=st["distinct"], units=st["units"], unit_a=st["unit_a"], unit_b=st["unit_b"])
            nA = sum(c["nA"] for c in ctas)
            an["phaseA_tbs"] = nA * UNIT / (st["A"] * 1e-6) / 1e12 / 1 if st["A"] else 0
            print("REC " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in an.items()
                                        if k not in ("snaps", "stall_at", "latA", "latB")}), flush=True)
            print("REC   snaps " + json.dumps(an["snaps"]) + " latA(med,p90,sum) " + str([round(v, 3) for v in an["latA"]])
                  + " latB " + str([round(v, 3) for v in an["latB"]]), flush=True)
            print("REC   wait(count, per CTA us, median) " + json.dumps(
                {k: [v[0], round(v[1], 3), round(v[2], 3)] for k, v in an["stall_at"].items()}), flush=True)
            if rp is not None:
                base = w_[10].view(torch.uint8).reshape(-1)
                lists = [[u["blk"] for u in c["iss"][:c["nA"]]] for c in ctas]
                for dl in [int(x) for x in a.replay.split(",")]:
                    med, _ = rpb.run(rp, base, flush, lambda r, L=lists: L, 3, dl, 5)
                    print(f"REPLAY same-bytes {an['label']} delay {dl} span {med['span_us']:.2f} TB/s {med['tbs']:.3f}"
                          f" steady {med['steady_tbs']:.3f} cta_unit {med['cta_unit_us']:.3f} lat {med['lat_med_us']:.2f}"
                          f"/{med['lat_p90_us']:.2f}", flush=True)
                d = w_[11].data_ptr() - w_[10].data_ptr()
                if d > 0 and d % UNIT == 0:
                    ab = [[u["blk"] if not u["pb"] else d // UNIT + u["blk"] for u in c["iss"][:c["nA"] + c["nB"]]]
                          for c in ctas]
                    # st_replay takes only the base pointer: B units sit d bytes past w13's start (another allocation)
                    med, _ = rpb.run(rp, base, flush, lambda r, L=ab: L, 3, 0, 5, graph=False)
                    print(f"REPLAY same-bytes-AB {an['label']} eager event {med['event_us']:.2f} span {med['span_us']:.2f}"
                          f" TB/s {med['tbs']:.3f} steady {med['steady_tbs']:.3f} vs helper0 plain {an['plain_us']:.2f}"
                          f" span {an['span']:.2f}", flush=True)
                else:
                    print(f"REPLAY AB skipped: w13 / w2 offset {d}", flush=True)
            res.append(an)
            if j < 2:
                dump.append(dict(label=an["label"], ctas=ctas))
        if lsc:
            lsc.save()
    if a.out:
        json.dump(dict(dump=dump), open(a.out, "w"))


if __name__ == "__main__":
    main()
