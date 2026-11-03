"""Write a self-contained short-load paired bench for muse_moe_w4a16 from the public release bytes.

    python3 tools/moe_short_bench.py --cand CAND.py --define -DOKC_X=1 [--define ...] --name NAME --out OUT.py
        [--base muse_moe_w4a16/cand/current_control_release.py] [--reps 48] [--sizes 1,8,16]

OUT.py is the release (5df257af) byte for byte, followed by one appended block. The judged path (prepare's
state, kernel_fn) is the release's, so the official score of a check is a control draw. The block wraps
prepare(): after the release's own prepare returns, it builds the candidate's Tiny / Tiny2 modules from the
helper0 source of CAND.py plus the --define flags, and runs three arms in-process on the judge's weights:

  ctrl   the release's modules on a fresh scratch set
  ctrl2  the release's modules on a second fresh scratch set (the null arm)
  cand   the candidate's Tiny / Tiny2 modules on a third scratch set (MID and larger stay the release's)

Each arm gets its own CUDA graph per size over static input buffers. Inputs are held-out routings from a copy of
reference.generate_input (dist normal, rank busiest, seeds from SEED0), one fresh input per repetition, copied into
the static buffers before the replay. Every replay starts after a read of a 256 MB buffer (cold L2). Arms are
interleaved, with the order rotated and reversed so that every permutation appears equally often. Every replay's
output is kept and checked against an FP32 reference computed from the judge's weights (relative L2 at most 1.5e-2,
every row at most 5x that, finite), and the candidate's output is compared with the control's.

The bench never fails prepare: errors print as MOE_SB error and the release's state is returned unchanged. It
stops starting new work once prepare has run SB_BUDGET_S seconds.
"""
import argparse
import hashlib
from pathlib import Path

MARK = "_MA_ALT_SRCS = {'moe_helper0': r'''"
END = "\n'''}\n"
BASE_PIN = '5df257afd6ae2863bc706826a9798d97b17e73830f1d10703c81bba21203eb9e'

BLOCK = r'''

# ---- short-load paired bench (tools/moe_short_bench.py); the judged path above is the release's
_MA_ALT_SRCS['moe_helper0_sb'] = r__SBQ____SB_HELPER____SBQ__
SB_NAME = __SB_NAME__
SB_EXTRA = __SB_EXTRA__
SB_HELPER_SHA256 = __SB_HELPER_SHA__
SB_SIZES = __SB_SIZES__
SB_REPS = __SB_REPS__
SB_SEED0 = 910001
SB_BUDGET_S = 280.0
SB_FLUSH_BYTES = 256 << 20
SB_ARMS = ('ctrl', 'ctrl2', 'cand')
SB_CAND_KEYS = ('tiny', 'tiny2')


def _sb_defines(key):
    base = {'tiny': MOE_TINY_DEFINES, 'tiny2': MOE_TINY2_DEFINES}[key]
    out = tuple('@moe_helper0_sb' if d == '@moe_helper0' else d for d in base) + tuple(SB_EXTRA)
    if out.count('@moe_helper0_sb') != 1:
        raise RuntimeError('SB_DEFINES')
    return out


def _sb_state(state, cand):
    st = dict(state)
    st['guards'] = []
    st['ma_ext'] = dict(state['ma_ext'])
    st['ma_scratch'] = [_expert_moe_guarded(t, st) for t in _ma_make_scratch(state['device'])]
    if cand:
        sfx = tuple(state['ma_sfx'])
        for key in SB_CAND_KEYS:
            st['ma_ext'][key] = _RouterExt(_ma_ext(_sb_defines(key) + sfx))
    return st


def _sb_launch(st, tokens, x, ids, tw, out):
    _fork_native_moe(tokens, st).okc_run(st['ma_w'], st['ma_scratch'], x, ids, tw, out, 0, 32, st['nsm'])


def _sb_input(bs, seed, device):
    """reference.generate_input(bs, 'ep8', seed), dist normal, rank busiest."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=device)
    z = torch.randn((256,), generator=gen, device=device)
    u = torch.rand((256,), generator=gen, device=device).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    tw = (routed.values / routed.values.sum(-1, keepdim=True)).float()
    return x.contiguous(), ids.to(torch.int32).contiguous(), tw.contiguous()


def _sb_dequant(packed, sf, gs):
    p = packed.view(torch.uint8)
    codes = torch.stack((p & 15, p >> 4), dim=-1).flatten(-2)
    grid = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=torch.float32, device=p.device)
    mag = grid[(codes & 7).long()]
    v = torch.where((codes & 8) != 0, -mag, mag)
    k = v.shape[-1]
    scale = sf.float() * torch.as_tensor(gs, dtype=torch.float32, device=p.device)
    return (v.view(*v.shape[:-1], k // 16, 16) * scale.unsqueeze(-1)).view_as(v)


class _SbRef:
    def __init__(self, weights):
        self.w = weights
        self.cache = {}
        self.nw = weights['post_norm_w'].float()

    def expert(self, e):
        if e not in self.cache:
            w = self.w
            self.cache[e] = (_sb_dequant(w['w13'][e], w['w13_sf'][e], w['w13_gs'][e]),
                             _sb_dequant(w['w2'][e], w['w2_sf'][e], w['w2_gs'][e]))
        return self.cache[e]

    def __call__(self, x, ids, tw):
        xf = x.float()
        out = torch.zeros(xf.shape, dtype=torch.float32, device=xf.device)
        il = ids.long()
        for e in torch.unique(il).tolist():
            if not 0 <= e < 32:
                continue
            tok, slot = (il == e).nonzero(as_tuple=True)
            w13, w2 = self.expert(e)
            h = xf[tok] @ w13.T
            inter = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
            y = (inter @ w2.T) * (1.0 + self.nw)
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
            out.index_add_(0, tok, y * tw[tok, slot].unsqueeze(1))
        return out.to(torch.bfloat16)


def _sb_err(out, ref):
    """(relative L2, worst row ratio, finite) as checks.compare forms them."""
    o, r = out.float(), ref.float()
    d = o - r
    rel = d.norm() / r.norm().clamp(min=1e-12)
    rows_ref = r.norm(dim=-1)
    floor = (0.1 * rows_ref.mean()).clamp(min=1e-12)
    row = (d.norm(dim=-1) / torch.maximum(rows_ref, floor)).max()
    return torch.stack((rel, row, torch.isfinite(o).all().float()))


def _sb_median_ci(d):
    """Median and the distribution-free 95% interval (order statistics of the binomial)."""
    import math
    d = sorted(d)
    n = len(d)
    med = d[n // 2] if n % 2 else 0.5 * (d[n // 2 - 1] + d[n // 2])
    k, acc = 0, 0.0
    while k < n:
        p = math.comb(n, k) / 2.0 ** n
        if acc + p > 0.025:
            break
        acc += p
        k += 1
    lo, hi = (d[k - 1], d[n - k]) if 0 < k <= n - k else (d[0], d[-1])
    return med, lo, hi


def _sb_order(r):
    k = r % 3
    o = list(range(3))[k:] + list(range(3))[:k]
    return o[::-1] if (r // 3) % 2 else o


def _sb_run(weights, state, t0):
    import time as _t
    dev = state['device']
    left = lambda: SB_BUDGET_S - (_t.monotonic() - t0)
    tc = _t.monotonic()
    arms = [_sb_state(state, False), _sb_state(state, False), _sb_state(state, True)]
    torch.cuda.synchronize(dev)
    print('MOE_SB %s extra=%s helper=%s sizes=%s reps=%d prep_s=%.1f cand_build_s=%.1f'
          % (SB_NAME, ' '.join(SB_EXTRA), SB_HELPER_SHA256[:16], ','.join(map(str, SB_SIZES)), SB_REPS,
             tc - t0, _t.monotonic() - tc), flush=True)
    if left() < 15:
        print('MOE_SB skip: %.1f s left after the builds' % left(), flush=True)
        return
    ref = _SbRef(weights)
    inputs, refs, nloc = {}, {}, {}
    for T in SB_SIZES:
        inputs[T] = [_sb_input(T, SB_SEED0 + 1000 * T + r, dev) for r in range(SB_REPS)]
        refs[T] = [ref(*inp) for inp in inputs[T]]
        nloc[T] = [int(torch.unique(inp[1][(inp[1] >= 0) & (inp[1] < 32)]).numel()) for inp in inputs[T]]
        if left() < 10:
            print('MOE_SB skip: %.1f s left after the references' % left(), flush=True)
            return
    ref.cache.clear()
    flush = torch.ones(SB_FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    sink = torch.zeros((), dtype=torch.float32, device=dev)
    graphs, bufs, outs, evs = {}, {}, {}, {}
    for T in SB_SIZES:
        for a, st in enumerate(arms):
            x, ids, tw = (t.clone() for t in inputs[T][0])
            out = torch.empty_like(x)
            s = torch.cuda.Stream(device=dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                for _ in range(2):
                    _sb_launch(st, T, x, ids, tw, out)
            torch.cuda.current_stream(dev).wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _sb_launch(st, T, x, ids, tw, out)
            graphs[T, a], bufs[T, a] = g, (x, ids, tw, out)
            outs[T, a] = torch.empty((SB_REPS,) + tuple(x.shape), dtype=x.dtype, device=dev)
    torch.cuda.synchronize(dev)
    done = 0
    for r in range(SB_REPS):
        if left() < 3:
            break
        for T in SB_SIZES:
            for a in _sb_order(r):
                x, ids, tw, out = bufs[T, a]
                xi, ii, wi = inputs[T][r]
                x.copy_(xi); ids.copy_(ii); tw.copy_(wi)
                torch.sum(flush, dim=0, out=sink)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[T, a].replay()
                e1.record()
                outs[T, a][r].copy_(out)
                evs[T, a, r] = (e0, e1)
        done += 1
    torch.cuda.synchronize(dev)
    print('MOE_SB timed reps=%d of %d, %.1f s left' % (done, SB_REPS, left()), flush=True)
    for T in SB_SIZES:
        tm = [[1000.0 * evs[T, a, r][0].elapsed_time(evs[T, a, r][1]) for r in range(done)] for a in range(3)]
        errs = [torch.stack([_sb_err(outs[T, a][r], refs[T][r]) for r in range(done)]).cpu() for a in range(3)]
        diff = torch.stack([(outs[T, 2][r].float() - outs[T, 0][r].float()).abs().max() for r in range(done)]).cpu()
        same = sum(int(torch.equal(outs[T, 2][r], outs[T, 0][r])) for r in range(done))
        nl = nloc[T][:done]
        parts = []
        for a in range(3):
            v = sorted(tm[a])
            parts.append('%s med %.2f min %.2f' % (SB_ARMS[a], v[len(v) // 2], v[0]))
        print('MOE_SB T=%d local experts %d..%d | %s' % (T, min(nl), max(nl), ' | '.join(parts)), flush=True)
        parts = []
        for name, (a, b) in (('cand-ctrl', (2, 0)), ('ctrl2-ctrl', (1, 0)), ('cand-ctrl2', (2, 1))):
            d = [tm[a][r] - tm[b][r] for r in range(done)]
            med, lo, hi = _sb_median_ci(d)
            parts.append('%s %+.2f [%+.2f,%+.2f] mean %+.2f neg %d/%d' % (name, med, lo, hi, sum(d) / len(d),
                                                                         sum(1 for v in d if v < 0), len(d)))
        print('MOE_SB T=%d d ' % T + ' | '.join(parts), flush=True)
        parts = []
        for a in range(3):
            e = errs[a]
            ok = int(((e[:, 0] <= 1.5e-2) & (e[:, 1] <= 7.5e-2) & (e[:, 2] > 0)).sum())
            parts.append('%s rel %.2e row %.2e pass %d/%d' % (SB_ARMS[a], float(e[:, 0].max()), float(e[:, 1].max()), ok, done))
        print('MOE_SB T=%d num %s | cand-ctrl maxabs %.3g equal %d/%d' % (T, ' | '.join(parts), float(diff.max()), same, done),
              flush=True)
    for a, st in enumerate(arms):
        _expert_moe_check_guards(st)
    print('MOE_SB guards ok', flush=True)
    del graphs, bufs, outs, evs, flush, refs, inputs, arms


_sb_release_prepare = prepare


def prepare(weights):
    import time as _t
    t0 = _t.monotonic()
    state = _sb_release_prepare(weights)
    try:
        _sb_run(weights, state, t0)
    except Exception as exc:
        import traceback
        print('MOE_SB error %r\n%s' % (exc, traceback.format_exc()[-1500:]), flush=True)
    torch.cuda.synchronize(state['device'])
    torch.cuda.empty_cache()
    return state
'''


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def helper_of(text):
    assert text.count(MARK) == 1
    body = text.split(MARK, 1)[1]
    assert END in body
    return body.split(END, 1)[0]


def build(base_text, cand_text, defines, name, reps, sizes):
    assert sha(base_text) == BASE_PIN, 'generator is pinned to the release 5df257af'
    helper = helper_of(cand_text)
    assert "'''" not in helper and '__SB' not in helper
    tail_base = base_text.split(MARK, 1)[1].split(END, 1)[1]
    tail_cand = cand_text.split(MARK, 1)[1].split(END, 1)[1]
    assert tail_base == tail_cand, 'the candidate changes host code after helper0; this bench only swaps helper0'
    block = BLOCK
    for key, value in (('__SBQ__', "'''"), ('__SB_HELPER__', helper), ('__SB_NAME__', repr(name)),
                       ('__SB_EXTRA__', repr(tuple(defines))), ('__SB_HELPER_SHA__', repr(sha(helper))),
                       ('__SB_SIZES__', repr(tuple(sizes))), ('__SB_REPS__', repr(reps))):
        block = block.replace(key, value)
    return base_text + block, sha(helper)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', type=Path, default=Path('muse_moe_w4a16/cand/current_control_release.py'))
    ap.add_argument('--cand', type=Path, required=True)
    ap.add_argument('--define', action='append', default=[])
    ap.add_argument('--name', required=True)
    ap.add_argument('--reps', type=int, default=48)
    ap.add_argument('--sizes', default='1,8,16')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    text, helper_sha = build(args.base.read_text(), args.cand.read_text(), args.define, args.name, args.reps,
                             [int(s) for s in args.sizes.split(',')])
    import ast
    ast.parse(text)
    args.out.write_text(text)
    print('wrote', args.out, len(text.encode()), 'bytes sha256', sha(text), 'helper', helper_sha)
