"""Appended unchanged to the pinned adapter and offline-extracted input generator."""
import hashlib
import json
import random
import statistics

_H_NATIVE_LOADER = load_inline
_H_KEYS = ('hidden_states', 'topk_ids', 'topk_weights')


def _h_loader(**kwargs):
    descriptor = dict(kwargs)
    for key in ('cpp_sources', 'cuda_sources'):
        descriptor[key] = hashlib.sha256(descriptor[key].encode()).hexdigest()
    require(descriptor in H_BUILD_CALLS, 'exact compiler source/flags/name/exports')
    return _H_NATIVE_LOADER(**kwargs)


def _h_routes(data):
    ids, weights = data['topk_ids'], data['topk_weights']
    require(bool(((ids >= 0) & (ids < 256)).all()), 'global route ID range')
    ordered = ids.sort(dim=-1).values
    require(bool((ordered[:, 1:] != ordered[:, :-1]).all()), 'distinct top8 experts')
    require(bool(torch.isfinite(weights).all()) and bool((weights >= 0).all())
            and bool(((weights.sum(-1) - 1).abs() <= 2e-6).all()), 'route normalization')
    require(bool(torch.isfinite(data['hidden_states']).all()), 'finite inputs')
    counts = torch.bincount(ids.flatten().long(), minlength=256)[:32]
    require(int(counts.max()) <= data['config']['bs'], 'per-expert capacity')
    return [int(counts.sum()), int((counts > 0).sum()), int(counts.max())]


def _h_errors(output, reference):
    require(output.shape == reference.shape and output.dtype == reference.dtype == torch.bfloat16,
            'output metadata')
    actual, expected = output.float(), reference.float()
    require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), 'nonfinite output')
    delta = actual - expected
    global_error = float(delta.norm() / expected.norm().clamp_min(1e-12))
    row_norm = expected.norm(dim=-1)
    floor = (.1 * row_norm.mean()).clamp_min(1e-12)
    row_error = float((delta.norm(dim=-1) / torch.maximum(row_norm, floor)).max())
    require(global_error <= GLOBAL_TOL and row_error <= ROW_TOL, 'paired output error')
    require(not bool((actual[row_norm == 0] != 0).any()), 'zero-reference rows must be exact zero')
    return [global_error, row_error]


def _h_ownership(states):
    seen, result = set(), []
    require(len({id(state['ma_ext']) for state in states}) == 4, 'private extension maps')
    for index, state in enumerate(states):
        require(state['_expert_pair_arm'] is (index >= 2), 'placement binding')
        owned, byte_count = set(), 0
        for tensor in [*state['ma_w'], *(full for full, words in state['guards'])]:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            require(key not in seen and key not in owned, 'private weight/scratch storage')
            owned.add(key)
            byte_count += storage.nbytes()
        require(len(state['ma_scratch']) == len(state['guards']), 'scratch guard inventory')
        for tensor, (full, words) in zip(state['ma_scratch'], state['guards']):
            require(tensor.untyped_storage().data_ptr() == full.untyped_storage().data_ptr()
                    and tensor.data_ptr() == full.data_ptr() and tensor.numel() * tensor.element_size() <= words * 4,
                    'scratch guard owner')
        seen.update(owned)
        result.append(byte_count)
    return result


def _h_reset(states):
    failure = None
    for state in states:
        try:
            initial = _ma_make_scratch(state['device'])
            require(len(initial) == len(state['ma_scratch']), 'scratch reset length')
            for target, source in zip(state['ma_scratch'], initial):
                require(target.shape == source.shape and target.dtype == source.dtype, 'scratch reset metadata')
                target.copy_(source)
            del initial
            torch.cuda.synchronize(state['device'])
            _expert_moe_check_guards(state)
        except Exception as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


def _h_case(states, seed, tokens, dist, flush, deadline, counters):
    device = states[0]['device']
    data = generate_input(tokens, 'ep8', seed, dist=dist, device=device)
    census = _h_routes(data)
    saved = {key: data[key].clone() for key in _H_KEYS}
    pointers = tuple(data[key].data_ptr() for key in _H_KEYS)
    captured = [dict(config=dict(data['config']), **{key: data[key].clone() for key in _H_KEYS}) for state in states]
    captured_pointers = [tuple(item[key].data_ptr() for key in _H_KEYS) for item in captured]
    require(len({pointer for group in captured_pointers for pointer in group}) == 12, 'private graph inputs')
    graphs, outputs = [], []
    try:
        for index, state in enumerate(states):
            _expert_moe_run(captured[index], state, index >= 2)
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with torch.cuda.graph(graph):
                outputs.append(_expert_moe_run(captured[index], state, index >= 2))
            counters['captures'] += 1
        require(len({output.untyped_storage().data_ptr() for output in outputs}) == 4, 'independent outputs')
        require(not {output.data_ptr() for output in outputs}.intersection(
                pointer for group in captured_pointers for pointer in group), 'output/input alias')

        def observe(empty=False):
            _expert_moe_budget(deadline)
            _h_routes(data)
            require(tuple(data[key].data_ptr() for key in _H_KEYS) == pointers, 'captured input pointers')
            for item, expected_pointers in zip(captured, captured_pointers):
                require(tuple(item[key].data_ptr() for key in _H_KEYS) == expected_pointers, 'private captured pointers')
                for key in _H_KEYS:
                    item[key].copy_(data[key])
            references = [_expert_moe_run(data, states[index], False).clone() for index in (0, 1)]
            _h_errors(references[1], references[0])
            for output, graph in zip(outputs, graphs):
                output.fill_(17)
                graph.replay()
            torch.cuda.synchronize(device)
            for output in outputs:
                for reference in references:
                    errors = _h_errors(output, reference)
                    counters['worst'] = [max(old, new) for old, new in zip(counters['worst'], errors)]
                if empty:
                    require(int(torch.count_nonzero(output)) == 0, 'empty routes output')
            for state in states:
                _expert_moe_check_guards(state)
            counters['observations'] += 1

        observe()
        fresh = generate_input(tokens, 'ep8', seed + 97, dist=dist, device=device)
        for key in _H_KEYS:
            data[key].copy_(fresh[key])
        observe()
        for transition in ('empty', 'occupied', 'empty', 'occupied'):
            first = 32 if transition == 'empty' else 0
            ids = torch.arange(first, first + 8, device=device, dtype=torch.int32)
            data['topk_ids'].copy_(ids[None, :].expand(tokens, 8))
            observe(transition == 'empty')
        for key in _H_KEYS:
            data[key].copy_(saved[key])
        observe()
        timings = None
        if tokens in LOADS:
            samples = [[] for state in states]
            rng = random.Random(seed + tokens + (1000 if dist == 'skewed' else 0))
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for repeat in range(REPEATS // 2):
                order = list(range(4))
                rng.shuffle(order)
                for traversal in (order, list(reversed(order))):
                    for index in traversal:
                        _expert_moe_budget(deadline)
                        flush.add_(1)
                        start.record()
                        graphs[index].replay()
                        end.record()
                        end.synchronize()
                        samples[index].append(positive(start.elapsed_time(end) * 1000))
                        counters['timed_replays'] += 1
            timings, ratios, nulls = paired_samples(samples)
            for output in outputs:
                for index in (0, 1):
                    _h_errors(output, _expert_moe_run(data, states[index], False))
        return [seed, tokens, dist, timings, census, ratios, nulls] if timings is not None else None
    finally:
        for key in _H_KEYS:
            data[key].copy_(saved[key])
        for item, expected_pointers in zip(captured, captured_pointers):
            for key in _H_KEYS:
                item[key].copy_(saved[key])
            require(tuple(item[key].data_ptr() for key in _H_KEYS) == expected_pointers
                    and all(torch.equal(item[key], saved[key]) for key in _H_KEYS), 'private input restoration')
        torch.cuda.synchronize(device)
        for graph in graphs:
            graph.reset()
        require(tuple(data[key].data_ptr() for key in _H_KEYS) == pointers
                and all(torch.equal(data[key], saved[key]) for key in _H_KEYS), 'input restoration')
        counters['restorations'] += 1


def prepare(weights):
    global load_inline
    require(not H_HOLD, 'HOLD: parent review and explicit captain allocation required')
    began = time.monotonic()
    deadline = began + 295
    states = []
    original_loader = load_inline
    counters = dict(observations=0, restorations=0, captures=0, timed_replays=0, worst=[0., 0.])
    rows = []
    report = None
    try:
        load_inline = _h_loader
        for arm in (False, False, True, True):
            states.append(_expert_moe_prepare(weights, arm, deadline))
        owners = _h_ownership(states)
        require(all(state['ma_sc6'] and tuple(state['ma_sfx']) == ('-DOKC_PRMTSC=1',) for state in states),
                'pinned SC6/PRMT route only; no fallback')
        flush = torch.zeros(64 * 1024 * 1024, dtype=torch.int32, device=states[0]['device'])
        for seed in BANKS:
            for dist in DISTRIBUTIONS:
                for tokens in TOKENS:
                    row = _h_case(states, seed, tokens, dist, flush, deadline, counters)
                    if row is not None:
                        rows.append(row)
        report = dict(runtime_id=H_RUNTIME_ID, official_arm='candidate2', placement_arms=[False, False, True, True],
                      owners_bytes=owners, shared_tensor_storage=[], rows=rows, counters=counters)
    finally:
        load_inline = original_loader
        _h_reset(states)
    _expert_moe_budget(deadline)
    report.update(prepare_s=time.monotonic() - began, cleanup_restored=True)
    report['component_pass'] = decision(rows)['component_pass']
    encoded = json.dumps(report, separators=(',', ':'), allow_nan=False)
    require(len(encoded.encode()) < CAPSULE_LIMIT, 'bounded capsule')
    _expert_moe_budget(deadline)
    print('EXPERT_MOE_CAPSULE ' + encoded, flush=True)
    return states[2]


def kernel_fn(data, state):
    return _expert_moe_run(data, state, True)
