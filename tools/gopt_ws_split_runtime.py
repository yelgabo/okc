"""Metadata validation and bounded failure reporting embedded in the private endpoint."""
_WS_LIVE = {}


def _ws_positions(ids):
    counts = [0] * 32
    for value in ids:
        if 0 <= value < 32:
            counts[value] += 1
    offsets, total = [], 0
    for count in counts:
        offsets.append(total)
        total += count
    result = []
    for value in ids:
        if 0 <= value < 32:
            result.append(offsets[value])
            offsets[value] += 1
        else:
            result.append(-1)
    return result


def _ws_validate_metadata(states, data):
    if data['config']['bs'] > 16:
        return
    expected = _ws_positions(data['topk_ids'].flatten().cpu().tolist())
    for state in states[2:]:
        meta = state['ma_scratch'][3]
        actual = meta.view(torch.int16)[:len(expected)].cpu().tolist()
        require(actual == expected, 'WS exported route map')
        gen = int(meta[64])
        ctr = state['ma_scratch'][0]
        require(int(ctr[3]) == gen + 1, 'WS latched generation advanced once')
        require(not bool(ctr[:2].any()) and not bool(ctr[8:96].any()), 'WS finalizer resets')
        current = state['ma_scratch'][1][:3072] if gen & 1 else ctr[1032:1032+3072].view(torch.float32)
        require(not bool(current.any()), 'WS alternate ss remains cleared')


def _ws_encode(report):
    import base64
    import zlib
    return base64.b64encode(zlib.compress(json.dumps(report, separators=(',', ':'), allow_nan=False).encode(), 9)).decode()


def _ws_partial(error):
    report = {key: value for key, value in _WS_LIVE.items() if key != 'primary_error'}
    report.update(runtime_id=H_RUNTIME_ID, status='FAILED_PARTIAL_NOT_QUALIFIED',
                  elapsed_s=time.monotonic()-report['began'], error=type(error).__name__ + ':' + str(error)[:300])
    blob = _ws_encode(report)
    require(len(blob) < 12000, 'bounded failure evidence')
    print('GOPT_WS_SPLIT_PARTIAL ' + blob, flush=True)


def _ws_probe(states, deadline, counters):
    device = states[2]['device']
    for tokens in (1, 8, 16):
        for gen in (0, 1):
            _expert_moe_budget(deadline)
            meta = torch.zeros(128, dtype=torch.int32, device=device)
            positions = [i if i % 3 else -1 for i in range(tokens*8)]
            meta.view(torch.int16)[:tokens*8].copy_(torch.tensor(positions, dtype=torch.int16, device=device))
            meta[64] = gen
            rng = torch.Generator(device=device).manual_seed(127+tokens+gen)
            y = torch.randn((tokens*8,4096), generator=rng, dtype=torch.float32, device=device)
            if gen == 0:
                y.zero_()
            ss = torch.zeros((2,3072), dtype=torch.float32, device=device)
            ss[gen,:tokens*8].copy_((y*y).sum(-1))
            tw = torch.rand((tokens,8), generator=rng, dtype=torch.float32, device=device)
            tw.div_(tw.sum(-1,keepdim=True))
            expected = torch.zeros((tokens,4096), dtype=torch.float32, device=device)
            for slot in range(8):
                q = torch.tensor(positions[slot::8], dtype=torch.int64, device=device)
                index = q.clamp_min(0)
                scale = tw[:,slot] * torch.rsqrt(ss[gen,index]*(1./4096)+1e-8)
                expected.add_(torch.where((q>=0)[:,None], scale[:,None]*y[index], 0.))
            output = torch.full_like(expected, 17, dtype=torch.bfloat16)
            ctr = torch.full((8,), -1, dtype=torch.int32, device=device)
            done = torch.full((88,), -1, dtype=torch.int32, device=device)
            _fork_native_moe(tokens,states[2]).okc_ws_finalize_probe([meta,ss,y,tw,output,ctr,done])
            torch.cuda.synchronize(device)
            _h_errors(output, expected.to(torch.bfloat16))
            require(int(ctr[3]) == gen+1 and not bool(ctr[:2].any()) and not bool(done.any()), 'WS probe reset')
            counters['finalizer_reference_cases'] = counters.get('finalizer_reference_cases',0)+1
