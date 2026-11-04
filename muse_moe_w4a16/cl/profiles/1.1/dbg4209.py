"""CL-1.1 debug: where the kernel's inter differs from helper0 for one input (default seed 4209, T = 32)."""
import os
import sys

sys.path.insert(0, "/root/okc/muse_moe_w4a16/cl/tests")
sys.path.insert(0, "/root/okc/muse_moe_w4a16/cl")
sys.path.insert(0, "/root/okc/muse_moe_w4a16")
sys.path.insert(0, "/root/okc/muse_moe_w4a16/cl/tests/helper0_debug")
import torch
import test_phase_a as tp
import cl_phase_a_bench as pab
import run_unit_math as rum
import build_debug
import reference

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 4209
T = 32
ext = pab.build(tuple(sys.argv[2:]))
pub = pab.load_public()
h0 = build_debug.build(pub, rum.build_defines(pub)[T], pub._MA_SFX_IF_PRMT)
with pab.GpuLock("/root/okc/.build/box/gpu.lock"):
    weights = reference.generate_weights(reference.WEIGHTS_SEED)
    w13u, gs = pab.pack_w13(pub, weights)
    runner = pab.Runner(ext, w13u, gs)
    local = {k: weights[k][:32].contiguous() for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
    local["post_expert_norm"] = weights["post_norm_w"]
    st = {"ma_sc6": False, "guards": [], "dev": torch.device("cuda"), "nsm": 132}
    pub._fork_native_ma_prepare(local, st)
    data = reference.generate_input(bs=T, sharding="ep8", seed=seed)
    totals = []
    for rep in range(int(os.environ.get('REPS', '3'))):
        kin, d, unw = tp.kernel_inter(runner, data)
        hin = tp.helper0_inter(pub, rum, h0, st, data)
        tot = 0
        for e, (toks, kv) in kin.items():
            hv = hin[e][1]
            bad = (tp.bits(kv) != tp.bits(hv))
            n = int(bad.sum())
            tot += n
            if n:
                rows, cols = bad.nonzero(as_tuple=True)
                sl = [i for i in range(d["n_slices"]) if d["slice_expert"][i] == e]
                print(f"rep {rep} expert {e} ntok {len(toks)} slices {[(d['slice_col0'][i], d['slice_ncol'][i]) for i in sl]}"
                      f" bad {n} rows {sorted(set(rows.tolist()))} cols {sorted(set(cols.tolist()))[:40]}")
        print("rep", rep, "total mismatch", tot)
        totals.append(tot)
    print("SUMMARY reps", len(totals), "bad reps", sum(1 for t in totals if t), "elements", sum(totals))
