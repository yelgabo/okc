"""Generate diagnostic-only all12 owned/SGLang graph timelines; never submit.

No source compression, runtime file access, subprocesses, metric assumptions or
kernel modifications. Reference helpers are extracted offline from our established
paired harness. Both arms keep their own inputs and preparation state.
"""
import argparse
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ('_ab_inputs', '_ab_rms', '_ab_rope', '_ab_coeffs', '_ab_attention',
           '_ab_ref_front', '_ab_policy')

RUNTIME = r'''
_LT_ORIGINAL_PREPARE = prepare
_timeline = lambda state: None
_REF_STREAM_SEED, _REF_STREAM_MEAN = 5354, 0.7
_REF_STREAM_SHARED, _REF_STREAM_RANK, _REF_DRAFT_SHARE = 0.9, 4, 0.6


def _lt_busiest_start(counts):
    return LOCAL_EXPERTS * int(counts.argmax())


def _lt_reset(data, saved):
    data['stream'].copy_(saved[0])
    data['attn_input'].copy_(saved[1])


def _lt_collect(fn, state, data, flush, deadline):
    import statistics
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile
    saved = (data['stream'].clone(), data['attn_input'].clone())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for warm_idx in range(3):
            _lt_reset(data, saved)
            out = fn(data, state)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    _lt_reset(data, saved)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(data, state)
    torch.cuda.synchronize()
    if len(out) != 3 or tuple(out[0].shape) != (data['config']['tokens'], HIDDEN):
        raise RuntimeError('invalid graph output shape')
    if out[0].dtype != torch.bfloat16 or out[1].dtype != torch.float32:
        raise RuntimeError('invalid graph output dtype')
    if not all(bool(torch.isfinite(t).all()) for t in out):
        raise RuntimeError('nonfinite output')
    ids = out[2].long()
    local = (ids >= data['local_start']) & (ids < data['local_start'] + LOCAL_EXPERTS)
    route = {'local_slots': int(local.sum()),
             'local_slots_histogram': torch.bincount(local.sum(-1), minlength=TOP_K+1).tolist(),
             'unique_local': int(torch.unique(ids[local]).numel())}
    cold = []
    for replay_idx in range(5):
        if time.monotonic() > deadline:
            raise RuntimeError('timeline prepare deadline')
        _lt_reset(data, saved)
        flush.zero_()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        cold.append(round(start.elapsed_time(end) * 1000, 3))
    traces = []
    for replay_idx in range(3):
        if time.monotonic() > deadline:
            raise RuntimeError('timeline prepare deadline')
        _lt_reset(data, saved)
        flush.zero_()
        torch.cuda.synchronize()
        # A single replay only: the L2 flush is outside this profiler context.
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            graph.replay()
            torch.cuda.synchronize()
        events = []
        for ev in prof.profiler.kineto_results.events():
            if ev.device_type() == DeviceType.CUDA:
                start_ns, end_ns = _ev_ns(ev)
                events.append((start_ns, end_ns, ev.device_resource_id(), ev.name()))
        traces.append(summarize(events))
    ordered = sorted(traces, key=lambda t: t['span_us'])
    return {'cold_us': cold, 'cold_median_us': statistics.median(cold),
            'profile_spans_us': [t['span_us'] for t in traces],
            'trace': ordered[1], 'routing': route}


def prepare(weights):
    import types
    started = time.monotonic()
    deadline = started + 280
    state = _LT_ORIGINAL_PREPARE(weights)
    sg = types.ModuleType('okc_lt_sglang')
    exec(_LT_SGLANG_SOURCE, sg.__dict__)
    sg_weights = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in weights.items()}
    sg_state = sg.prepare(sg_weights)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device='cuda')
    report = {'schema': 1, 'pins': _LT_PINS, 'policy': _ab_policy(globals(), state),
              'counter_status': 'not collected', 'rows': []}
    shapes = [('d1', 1, 1), ('d16', 16, 1), ('d32', 32, 1),
              ('v1', 1, 8), ('v16', 16, 8), ('v32', 32, 8)]
    for shape_idx, (load, batch, length) in enumerate(shapes):
        for layer_idx, layer in enumerate(('sliding', 'full')):
            if time.monotonic() > deadline:
                raise RuntimeError('incomplete all12 timeline before deadline')
            seed = 87001 + shape_idx * 1000 + layer_idx * 100
            base = _ab_inputs(batch, length, layer, seed)
            tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                front = _ab_ref_front(base, weights)
                ids = front['biased'].topk(TOP_K, dim=-1).indices
                counts = torch.bincount((ids // LOCAL_EXPERTS).flatten(),
                                        minlength=N_EXPERTS // LOCAL_EXPERTS)
                start = _lt_busiest_start(counts)
                base['local_start'].fill_(start)
                reference_counts = counts.tolist()
                del front, ids, counts
            finally:
                torch.backends.cuda.matmul.allow_tf32 = tf32
            # Alternate acquisition order across shapes, retaining both independent inputs.
            arms = [('owned', kernel_fn, state), ('sglang', sg.kernel_fn, sg_state)]
            if (shape_idx + layer_idx) % 2:
                arms.reverse()
            for name, fn, arm_state in arms:
                data = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in base.items()}
                row = _lt_collect(fn, arm_state, data, flush, deadline)
                row.update(case=load+layer[0], arm=name, seed=seed, local_start=start,
                           reference_group_slots=reference_counts)
                report['rows'].append(row)
                del data
            del base
    if len(report['rows']) != 24:
        raise RuntimeError('missing timeline rows')
    report['prepare_s'] = time.monotonic() - started
    blob = encode_report(report)
    print('LAYER_TIMELINE_V1 ' + blob, flush=True)
    return state
'''


def extract_helpers():
    tree = ast.parse((ROOT / 'tools/dl_make_bench_pub.py').read_text(encoding='utf-8'))
    body = next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BODY'
                                                    for t in n.targets))
    parsed = ast.parse(body)
    found = {n.name: ast.get_source_segment(body, n) for n in parsed.body
             if isinstance(n, ast.FunctionDef) and n.name in HELPERS}
    if set(found) != set(HELPERS):
        raise ValueError('missing established reference helper')
    return '\n\n'.join(found[n] for n in HELPERS)


def build(candidate):
    baseline = (ROOT / 'muse_decode_layer/baseline.py').read_text(encoding='utf-8')
    codec = (ROOT / 'tools/dl_timeline_codec.py').read_text(encoding='utf-8')
    pins = {'candidate_lf_sha256': hashlib.sha256(candidate.encode()).hexdigest(),
            'sglang_lf_sha256': hashlib.sha256(baseline.encode()).hexdigest(),
            'input_helper': 'tools/dl_make_bench_pub.py: _ab_inputs + FP32-front busiest group'}
    result = (candidate + '\n\n' + codec + '\n\n' + extract_helpers()
              + '\n_LT_SGLANG_SOURCE = ' + repr(baseline)
              + '\n_LT_PINS = ' + repr(pins) + '\n' + RUNTIME)
    ast.parse(result)
    if len(result.encode()) >= 1_000_000:
        raise ValueError('diagnostic source exceeds judge payload limit')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--candidate', type=Path,
                        default=ROOT/'muse_decode_layer/cand/layer_attn31s.py')
    args = parser.parse_args()
    result = build(args.candidate.read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding='utf-8', newline='\n')
    print(len(result.encode()), hashlib.sha256(result.encode()).hexdigest(), args.output)
