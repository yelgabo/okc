"""Per-role GEMM time at verify 16 / 32 (m = 128 / 256): the layer file's pinned cuBLASLt plan against own BF16
kernels (the layer's bfp gemm90 / gq, and the BF16 board submission's tuned pick), each as a CUDA graph replayed
from a cold L2 with per-replay events, candidates interleaved. Standalone (one kernel per graph), so the numbers
rank kernels; edge gaps come from the layer timeline.

    python tools/bf16_role_times.py muse_decode_layer/cand/ly_stk3.py --router-seed S --lock /root/okc/.build/box/gpu.lock
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as bx  # noqa: E402
import torch  # noqa: E402

ROLE_LAYER = {"qkv": "qkv_gate", "o": "o", "un": "unpooler"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--bf16-sub", default="muse_bf16_proj/submission.py")
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    ap.add_argument("--reps", type=int, default=40)
    a = ap.parse_args()
    if a.lock:
        bx.release_lock_for_triton()
        bx.BudgetClock.install()
        bx.LOCK = bx.GpuLock(a.lock)
        bx.LOCK.acquire("start")
    bx.content_addressed_extensions(a.ext_cache)
    pdir = os.path.abspath("muse_decode_layer")
    sys.path.insert(0, pdir)
    ref = bx.load_module("reference", os.path.join(pdir, "reference.py"))
    torch.backends.cuda.matmul.allow_tf32 = False
    weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=a.router_seed)
    mod = bx.load_module("okc_submission", a.submission)
    if hasattr(mod, "_pdl8_setup"):
        mod._pdl8_setup = lambda st: None
    st = mod.prepare(weights)
    bst = st.get("bfp")
    print("ROLE bfp", None if bst is None else sorted(k for k in bst if not k.startswith("_"))[:30], flush=True)
    bmod = bx.load_module("bf16_sub", os.path.abspath(a.bf16_sub))
    bw = {k: weights[k].contiguous() if k != "router" else weights[k].float().contiguous()
          for k in ("qkv_gate", "o", "pooler", "unpooler", "router")}
    try:
        bstate = bmod.prepare(bw)
    except Exception as ex:  # noqa: BLE001
        print("ROLE bf16_sub prepare failed", repr(ex)[:300], flush=True)
        bstate = None
    torch.cuda.synchronize()
    dev = weights["o"].device
    flush = torch.ones(64 * 1024 * 1024 // 1024, 1024, dtype=torch.float32, device=dev)
    sink = torch.empty(1024, dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(77)

    def bsub(layer, x, m):
        n, k = bw[layer].shape
        return bmod.kernel_fn({"x": x, "config": {"layer": layer, "m": m, "n": n, "k": k}}, bstate)

    for m in (128, 256):
        for role in ("qkv", "o", "pr", "un"):
            plan = st["graft_lt"].get((role, m))
            if role == "pr":
                k = weights["pooler"].shape[1]
            else:
                k = weights[ROLE_LAYER[role]].shape[1]
            x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
            cands = {}
            if plan is not None:
                cands["lt"] = lambda x=x, plan=plan: mod._graft_lt_run(x, plan)
            if role != "pr":
                cands["torch"] = lambda x=x, lw=weights[ROLE_LAYER[role]]: torch.nn.functional.linear(x, lw)
            if role == "pr":
                pn, rn = weights["pooler"].shape[0], weights["router"].shape[0]
                if bst is not None:
                    cands["g90+g90r"] = lambda x=x: (mod.bfp__run_gemm(x, "pooler", m, pn, k, bst),
                                                      mod.bfp__run_gemm(x, "router", m, rn, k, bst))
                    if ("pooler", mod.bfp__gq_bn(m)) in mod.bfp_GQ_CONFIGS:
                        cands["gq+g90r"] = lambda x=x: (mod.bfp__run_gq(x, "pooler", m, pn, k, bst),
                                                         mod.bfp__run_gemm(x, "router", m, rn, k, bst))
                if bstate is not None:
                    cands["bsub_pool"] = lambda x=x: bsub("pooler", x, m)
                    cands["bsub_router"] = lambda x=x: bsub("router", x, m)
                    if bst is not None:
                        cands["bsub_pool+g90r"] = lambda x=x: (bsub("pooler", x, m),
                                                                mod.bfp__run_gemm(x, "router", m, rn, k, bst))
            else:
                layer = ROLE_LAYER[role]
                n = weights[layer].shape[0]
                if bst is not None:
                    cands["g90"] = lambda x=x, layer=layer, n=n: mod.bfp__run_gemm(x, layer, m, n, k, bst)
                    if (layer, mod.bfp__gq_bn(m)) in mod.bfp_GQ_CONFIGS:
                        cands["gq"] = lambda x=x, layer=layer, n=n: mod.bfp__run_gq(x, layer, m, n, k, bst)
                if bstate is not None:
                    cands["bsub"] = lambda x=x, layer=layer: bsub(layer, x, m)
            if bstate is not None:
                for lay in (("pooler", "router") if role == "pr" else (ROLE_LAYER[role],)):
                    pk = bstate.get("pick", {}).get((lay, m))
                    print(f"ROLE pick {lay} {m}: {getattr(pk, '__name__', None)} {getattr(pk, 'keywords', '')} "
                          f"{getattr(pk, 'args', '')}"[:200], flush=True)
            graphs = {}
            for name, fn in cands.items():
                try:
                    fn()
                    torch.cuda.synchronize()
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr):
                        fn()
                    torch.cuda.synchronize()
                    graphs[name] = gr
                except Exception as ex:  # noqa: BLE001
                    print(f"ROLE {role} {m} {name} failed {repr(ex)[:200]}", flush=True)
            names = list(graphs)
            evs = []
            for r in range(a.reps):
                order = names if r % 2 == 0 else names[::-1]
                k0 = r % len(names)
                order = order[k0:] + order[:k0]
                for nm in order:
                    torch.sum(flush, dim=0, out=sink)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[nm].replay()
                    e1.record()
                    evs.append((nm, e0, e1))
            torch.cuda.synchronize()
            per = {nm: [] for nm in names}
            for nm, e0, e1 in evs:
                per[nm].append(1000 * e0.elapsed_time(e1))
            base = statistics.median(per[names[0]]) if names else 0
            for nm in names:
                med = statistics.median(per[nm])
                print(f"ROLE m {m:3d} {role:3s} {nm:16s} {med:8.2f} us  vs {names[0]} {med - base:+7.2f}", flush=True)
            del graphs
    if bx.LOCK is not None:
        bx.LOCK.release("end")


if __name__ == "__main__":
    main()
