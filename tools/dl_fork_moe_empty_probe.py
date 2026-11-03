"""CHECK ONLY: call after native preparation; never use in a scored candidate.

The caller supplies the candidate's torch, selector and native factory. No input
cache is used: four fresh banks are copied between queued graph replays.
"""


def empty_conformance(torch, state, selector, native_factory, fallback_factory,
                      reps=2000, sizes=(32, 33, 128, 256)):
    import time
    started = time.monotonic()
    assert reps >= 1 and state['_fork_native_eligible'], 'Native path not exercised'
    dev = state['qkv'].device
    gen = torch.Generator(device=dev).manual_seed(612971)
    records = []

    def zero(out):
        assert out.dtype == torch.bfloat16
        assert bool(torch.isfinite(out).all()) and int(torch.count_nonzero(out)) == 0, 'EMPTY_NONZERO'

    def compare(out, ref):
        a, b = out.float(), ref.float()
        assert bool(torch.isfinite(a).all())
        diff = a - b
        global_err = float(diff.norm() / b.norm().clamp_min(1.e-8))
        row_err = float((diff.norm(dim=1) / b.norm(dim=1).clamp_min(1.e-8)).max())
        assert global_err <= .008 and row_err <= .075, ('OCCUPIED_MISMATCH', global_err, row_err)
        return global_err, row_err

    def run(ext, x, ids, tw, out):
        ext.okc_run(state['ma_w'], state['ma_scratch'], x, ids, tw, out, 0, 32, state['nsm'])

    for m in sizes:
        assert time.monotonic() - started < 280, 'EMPTY_PROBE_BUDGET'
        ext = selector(m, state)
        assert ext is native_factory(m, state), 'Native path not selected'
        helper = selector(1, state)
        assert helper is native_factory(1, state)
        control = fallback_factory(m, state)
        banks = torch.randn((4, m, 4096), device=dev, dtype=torch.bfloat16, generator=gen)
        x = banks[0].clone()
        ids = torch.arange(8, device=dev, dtype=torch.int32)[None, :].expand(m, 8).contiguous()
        empty = torch.full_like(ids, -1)
        tw = torch.full((m, 8), .125, device=dev, dtype=torch.float32)
        y, z = torch.empty_like(x), torch.empty_like(x)
        hx = x[:1].clone()
        hz = torch.empty_like(hx)
        hid, htw = empty[:1].contiguous(), tw[:1].contiguous()
        z.fill_(17)
        run(ext, x, empty, tw, z)
        torch.cuda.synchronize()
        zero(z)  # first explicit empty call starts from a nonzero output
        refs = []
        for bank in banks:
            ref = torch.empty_like(x)
            run(control, bank, ids, tw, ref)
            refs.append(ref)
        run(ext, x, ids, tw, y)
        torch.cuda.synchronize()
        compare(y, refs[0])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(ext, x, ids, tw, y)
            z.fill_(17)
            run(ext, x, empty, tw, z)
            hz.fill_(-23)
            run(helper, hx, hid, htw, hz)
        # Re-poison in EVERY replay, so a later no-write cannot reuse stale zero.
        # Three calls share native scratch and exercise both generation parities.
        y.fill_(float('nan')); z.fill_(17); hz.fill_(-23)
        for i in range(reps):
            x.copy_(banks[i % 4])
            graph.replay()
        torch.cuda.synchronize()  # launch/transfer faults are never optional
        zero(z); zero(hz)
        ge, re = compare(y, refs[(reps - 1) % 4])
        records.append((m, reps, ge, re, 'EAGER_AND_FINAL_REPLAY_EMPTY_EXACT_ZERO'))
        print('EMPTY_CONFORMANCE', records[-1], flush=True)
    print('EMPTY_CONFORMANCE_ALL', records, flush=True)
    return records
