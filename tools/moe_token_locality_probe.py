"""Append to the owned standalone source for a check-only token order experiment.

Test whether clustering token rows by their smallest local expert reduces gather
time enough to pay for a later implementation. All permutation work is outside
the timing, so this screens opportunity only.
"""
_locality_prepare = prepare


def prepare(weights):
    import math
    import statistics
    import random
    state = _locality_prepare(weights)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    rng = random.Random(1127)
    for seed in (781, 782):
        gains = [[], []]
        nulls = [[], []]
        for m in POLICY_BS:
            base = _make_inputs(m, seed + 1009*m)
            ids = base["topk_ids"]
            perm = ids.clamp_max(32).amin(dim=1).argsort(stable=True)
            identity = torch.arange(m, device="cuda")
            ref = kernel_fn(base, state).clone().float()
            graphs, retained, times = [], [], [[], [], [], []]
            for order in (identity, identity, perm, perm):
                data = {k: v.index_select(0, order) if isinstance(v, torch.Tensor) else v
                        for k, v in base.items()}
                kernel_fn(data, state)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = kernel_fn(data, state)
                graph.replay()
                torch.cuda.synchronize()
                expected = ref.index_select(0, order)
                rel = (output.float()-expected).norm()/expected.norm().clamp_min(1e-12)
                if not float(rel) <= 0.001:
                    raise RuntimeError(f"row equivalence failed {m} {seed} {float(rel)}")
                graphs.append(graph)
                retained.append((data, output))
            for rep in range(24):
                for arm in rng.sample(range(4), 4):
                    flush.zero_()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    times[arm].append(start.elapsed_time(end)*1000)
            med = [statistics.median(series) for series in times]
            for arm in range(2):
                gains[arm].append(med[arm]/med[2+arm])
                nulls[arm].append(med[2*arm]/med[2*arm+1])
            print("TOKEN_ORDER", seed, m, med, "moved", int((perm != identity).sum()), flush=True)
        print("TOKEN_GM", seed, [math.prod(s)**(1/6) for s in gains+nulls], flush=True)
    return state
