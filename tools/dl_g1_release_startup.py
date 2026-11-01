"""Bounded startup self-test for the single-arm Family G layer. No timing or selection."""
import argparse
import ast
import hashlib
import inspect
from pathlib import Path


BASE = '295ca590a9d07a2491d600e1c96ec5583d29ae3220f619bc9e99adf62ef48491'
CONTROL = '1607293eee3403109df5d37b6e2d1ee34b528c3578afa6c2c8668440afde1658'
POST_STAGE_FLUSH = '8195660dec57d9ceb6d6ab71d048fd5c144e657d8ec44022c75b0f34ca88bca1'


def _release_private_scratch(torch, buffers, sentinel):
    private, tails = [], []
    for original in buffers:
        size = original.numel() * original.element_size()
        words = (size + 3) // 4
        full = torch.full((words + 65536,), sentinel, dtype=torch.int32, device=original.device)
        view = full[:words].view(torch.uint8)[:size].view(original.dtype).view_as(original)
        view.copy_(original)
        private.append(view)
        tails.append(full[words:])
    return private, tails


def _release_routes(torch, size, device):
    rows = torch.arange(size, device=device, dtype=torch.int32)[:, None]
    cols = torch.arange(8, device=device, dtype=torch.int32)[None, :]
    start = 96
    regular = start + (rows + cols) % 32
    hot = (start + cols).expand(size, 8)
    partial = torch.where((cols == 0) & (rows % 3 != 1), start + rows % 4, start + 32 + cols)
    remote = (start + 32 + cols).expand(size, 8)
    return [torch.where((g >= start) & (g < start + 32), g - start, -1).contiguous()
            for g in (regular, hot, partial, remote)]


def _release_verify(torch, actual, expected):
    if actual.dtype != torch.bfloat16 or actual.shape != expected.shape:
        raise RuntimeError('G startup output shape or dtype')
    a, b = actual.float(), expected.float()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise RuntimeError('G startup nonfinite output')
    diff = a - b
    global_error = float(diff.norm() / b.norm().clamp_min(1e-8))
    row_error = float((diff.norm(dim=1) / b.norm(dim=1).clamp_min(1e-8)).max())
    if global_error > .008 or row_error > .075:
        raise RuntimeError(f'G startup expert comparison failed ({global_error}, {row_error})')


def _release_expert_self_test(torch, weights, production, selector, native_factory,
                              scratch_factory, marlin_prepare, sentinel,
                              sizes=(128, 32, 33, 256, 1, 8, 16), deadline=None):
    import time
    device = weights['w13'].device
    generator = torch.Generator(device=device).manual_seed(768921)
    private = dict(production)
    private['ma_ext'] = {}
    # Do not register private buffers in the layer's long-lived guard table.
    private['ma_scratch'], tails = _release_private_scratch(torch, scratch_factory(device), sentinel)
    production_storage = {x.untyped_storage().data_ptr() for x in production.get('ma_scratch', ())}
    if any(x.untyped_storage().data_ptr() in production_storage for x in private['ma_scratch']):
        raise RuntimeError('G startup mutable state alias')
    native = production['_fork_native_eligible']
    reference_state = {}
    marlin_prepare(weights, reference_state)
    reference = reference_state['_fork_marlin_adapter']
    try:
        if not native:
            # Preserve the real production fallback for supported non-native weight layouts.
            marlin_prepare(weights, private)
        helper = selector(1, private)
        checks = 0
        for size in sizes:
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError('G startup self-test exceeded prepare budget')
            ext = selector(size, private)
            if native and (ext is not native_factory(size, private) or helper is not native_factory(1, private)):
                raise RuntimeError('G startup native selection drift')
            routes = _release_routes(torch, size, device)
            banks = torch.randn((4, size, 4096), device=device, dtype=torch.bfloat16, generator=generator)
            scores = torch.randn((4, size, 8), device=device, dtype=torch.float32, generator=generator).softmax(-1)
            x, ids, tw = banks[0].clone(), routes[0].clone(), scores[0].clone()
            out, empty_out = torch.empty_like(x), torch.empty_like(x)
            helper_out = torch.empty_like(x[:1])
            empty_ids = routes[3]
            # The first WGS launch sees no local jobs and a newly allocated private scratch.
            empty_out.fill_(17)
            ext.okc_run(private['ma_w'], private['ma_scratch'], x, empty_ids, tw, empty_out,
                        0, 32, production['nsm'])
            torch.cuda.synchronize()
            if int(torch.count_nonzero(empty_out)):
                raise RuntimeError('G startup first empty route did not write zeros')
            refs = []
            for i in range(3):
                ref = torch.empty_like(x)
                reference.okc_run((), (), banks[i], routes[i], scores[i], ref, 0, 32, production['nsm'])
                refs.append(ref)
            refs.append(torch.zeros_like(x))

            def calls():
                out.fill_(float('nan'))
                ext.okc_run(private['ma_w'], private['ma_scratch'], x, ids, tw, out, 0, 32, production['nsm'])
                empty_out.fill_(17)
                ext.okc_run(private['ma_w'], private['ma_scratch'], x, empty_ids, tw, empty_out, 0, 32, production['nsm'])
                helper_out.fill_(-23)
                helper.okc_run(private['ma_w'], private['ma_scratch'], x[:1], empty_ids[:1], tw[:1],
                               helper_out, 0, 32, production['nsm'])

            calls()
            torch.cuda.synchronize()
            _release_verify(torch, out, refs[0])
            if int(torch.count_nonzero(empty_out)) or int(torch.count_nonzero(helper_out)):
                raise RuntimeError('G startup eager empty route did not write zeros')
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                calls()
            # Every replay is observed, with fresh inputs/IDs/weights at the captured pointers.
            for i in (3, 1, 2, 0, 2, 3, 0, 1):
                x.copy_(banks[i])
                ids.copy_(routes[i])
                tw.copy_(scores[i])
                graph.replay()
                torch.cuda.synchronize()
                _release_verify(torch, out, refs[i])
                if (i == 3 and int(torch.count_nonzero(out))) or int(torch.count_nonzero(empty_out)) or int(torch.count_nonzero(helper_out)):
                    raise RuntimeError('G startup replay empty route did not write zeros')
                checks += 1
            del graph
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError('G startup conformance incomplete within prepare budget')
        if any(bool((tail != sentinel).any()) for tail in tails):
            raise RuntimeError('G startup expert scratch overrun')
        print('G_RELEASE_STARTUP_OK', 'native' if native else 'Marlin', tuple(sizes), checks, 'verified_replays', flush=True)
    finally:
        reference_state.clear()
        private.clear()


WRAPPER = '''
_release_normal_prepare = prepare
def prepare(weights):
    import time
    deadline = time.monotonic() + 295
    state = _release_normal_prepare(weights)
    _release_expert_self_test(torch, weights, state, _moe, _fork_native_moe,
                              _ma_make_scratch, _fork_prepare_marlin, SCRATCH_SENTINEL, deadline=deadline)
    return state
'''


def _release_long_graph_test(torch, size, graph, x, ids, tw, banks, routes, scores,
                             out, empty_out, helper_out, refs, deadline, reps=2000):
    import time
    if reps < 4 or reps % 4:
        raise ValueError('expected complete routing-bank cycles')
    for i in range(reps):
        bank = i % 4
        if i % 128 == 0 and deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError('G1b4 long private replay budget')
        x.copy_(banks[bank])
        ids.copy_(routes[bank])
        tw.copy_(scores[bank])
        graph.replay()
    torch.cuda.synchronize()
    _release_verify(torch, out, refs[bank])
    if int(torch.count_nonzero(empty_out)) or int(torch.count_nonzero(helper_out)):
        raise RuntimeError('G1b4 empty output after long replay')
    # Also check occupied output immediately after the last all-remote replay.
    x.copy_(banks[0])
    ids.copy_(routes[0])
    tw.copy_(scores[0])
    graph.replay()
    torch.cuda.synchronize()
    _release_verify(torch, out, refs[0])
    if int(torch.count_nonzero(empty_out)) or int(torch.count_nonzero(helper_out)):
        raise RuntimeError('G1b4 empty helper failed after restoring occupied input')
    print('G1B4_PRIVATE_LONG_OK', size, reps, 'remote_and_restored_checked', flush=True)


def build(source, baseline_stress=False):
    supported = (CONTROL,) if baseline_stress else (BASE, POST_STAGE_FLUSH)
    if hashlib.sha256(source.encode()).hexdigest() not in supported:
        raise ValueError('unexpected pure layer for requested startup test')
    helpers = (_release_private_scratch, _release_routes, _release_verify, _release_expert_self_test)
    notice = '\n# Added by yelgabo, November 1, 2026: private-state startup validation.\n'
    validation = '\n\n'.join(inspect.getsource(f) for f in helpers)
    if baseline_stress:
        anchor = '            del graph\n'
        assert validation.count(anchor) == 1
        validation = validation.replace(anchor, '''            if native and size in (32, 33, 128, 256):
                _release_long_graph_test(torch, size, graph, x, ids, tw, banks, routes, scores,
                                         out, empty_out, helper_out, refs, deadline)
''' + anchor)
        validation += '\n\n' + inspect.getsource(_release_long_graph_test)
    output = source + notice + validation + '\n' + WRAPPER
    before = ast.parse(source).body
    after = ast.parse(output).body
    assert all(ast.dump(x) == ast.dump(y) for x, y in zip(before, after))
    assert all(isinstance(n, ast.FunctionDef) and n.name != 'kernel_fn' or
               isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == '_release_normal_prepare'
               for n in after[len(before):])
    payload = output.encode()
    if len(payload) >= 1_000_000:
        raise ValueError('exceeds one-file limit')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pure_stage', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline-stress', action='store_true',
                        help='build the G1b4 control with 2000 queued private replays at each of four sizes')
    args = parser.parse_args()
    data = build(args.pure_stage.read_text('utf8'), args.baseline_stress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(len(data), hashlib.sha256(data).hexdigest())
