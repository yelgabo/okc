"""Read-only pinned G0/607e precision inventory and receipt arithmetic. No GPU."""
import argparse, ast, hashlib, json, math, re, sys
from pathlib import Path


def audit(root):
    sys.path.insert(0, str(root / 'tools'))
    from dl_proj_budget_audit import load_times, counterfactual
    paths = [root / '.build/today/kel/kernels/muse_decode_layer.py',
             root / 'muse_decode_layer/cand/g1b4_stage_release.py']
    raw = [p.read_bytes() for p in paths]
    assert [hashlib.sha256(b).hexdigest() for b in raw] == [
        '1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a',
        'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab']
    sources = [b.decode().replace('\r\n', '\n') for b in raw]
    trees = [ast.parse(s) for s in sources]
    def constant(t, name):
        ns = [n for n in t.body if isinstance(n, ast.Assign)
              and any(getattr(x, 'id', '') == name for x in n.targets)]
        assert len(ns) == 1, name
        return ast.literal_eval(ns[0].value)
    literals = {}
    for name in ('ATT7_CUDA_SRC', 'ATT7_CPP_SRC', 'bfp_SK2_SRC'):
        a, b = [constant(t, name) for t in trees]
        literals[name] = dict(equal=a == b, bytes=[len(x.encode()) for x in (a, b)],
                              sha256=[hashlib.sha256(x.encode()).hexdigest() for x in (a, b)])
    assert literals['ATT7_CPP_SRC']['equal']
    a, b = [constant(t, 'ATT7_CUDA_SRC') for t in trees]
    b = b.replace('return __bfloat162float(__float2bfloat16_rn(x));', 'return x;', 1)
    b = b.replace('programmaticStreamSerializationAllowed = 0;', 'programmaticStreamSerializationAllowed = 1;')
    assert a == b, 'ATT7 delta exceeds pr8 BF16 round plus PDL host attribute'
    a, b = [constant(t, 'bfp_SK2_SRC') for t in trees]
    b = '\n'.join(x for x in b.split('\n') if 'upstream generic x -> TMA' not in x)
    b = b.replace('programmaticStreamSerializationAllowed = 0;', 'programmaticStreamSerializationAllowed = 1;')
    assert a == b, 'Ordinary SK2 delta exceeds explicit two proxy fences'
    files = ['muse_decode_layer.3a5f879df0d44825.json', 'g1-stage-release.607eb52bdaf24160.json']
    data = [json.loads((root / 'muse_decode_layer/profiles' / f).read_text(encoding='utf8')) for f in files]
    assert [d['job_id'] for d in data] == ['3a5f879df0d44825', '607eb52bdaf24160']
    times = [load_times(d) for d in data]
    bases = [load_times(d, 'baseline_ns') for d in data]
    def score(b, t):
        return math.exp(sum(math.log(b[k] / t[k]) for k in t) / 6)
    scores = [score(b, t) for b, t in zip(bases, times)]
    for s, d in zip(scores, data):
        assert abs(s - d['summary']['speedup']) < 1e-9
    rows = []
    savings = {k: {} for k in ('attention', 'dense', 'expert', 'glue')}
    for i, (o, n) in enumerate(zip(data[0]['benchmarks'], data[1]['benchmarks'])):
        key = (n['case']['load'], n['case']['layer'])
        assert key == (o['case']['load'], o['case']['layer'])
        groups = []
        for d in data:
            g = {k: 0. for k in savings}
            for line in d['profiles'][i]['report'].splitlines():
                m = re.match(r'(.+?)\s+([0-9.]+) µs\s+[0-9.]+%\s+(\d+)$', line)
                if not m:
                    continue
                name, t = m[1], float(m[2])
                k = 'attention' if ('attn_' in name or 'flash' in name) else 'expert' if 'moe_fused' in name else 'dense' if any(x in name for x in ('sk2_kernel', 'nvjet_', 'void rq<', '::router_kernel')) else 'glue'
                g[k] += t
            groups.append(g)
        for k in savings:
            savings[k][key] = max(0., groups[1][k] - groups[0][k])
        rows.append(dict(load=key[0], context=key[1], weight=n['case']['weight'],
                         g0_us=o['median_ns'] / 1000, g607_us=n['median_ns'] / 1000,
                         g0_profile_groups_us=groups[0], g607_profile_groups_us=groups[1]))
    return dict(source_pins=[dict(path=str(p), bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for p, b in zip(paths, raw)],
                literal_proof=literals, ordinary_sk2_delta_only_two_proxy_fences_and_pdl_attr=True, att7_delta_only_pr8_round_and_pdl_attr=True,
                scores=scores, score_ratio=scores[0] / scores[1], weighted_load_us=times,
                baseline_load_us=bases, common607_baseline_g0_score=score(bases[1], times[0]),
                short_positive_profile_gap_scenarios={k: counterfactual(data[1], {c: v for c, v in z.items() if c[0] in ('decode 1', 'verify 1')}) for k, z in savings.items()},
                per_group_positive_profile_gap_scenarios={k: counterfactual(data[1], v) for k, v in savings.items()},
                caveat='Cross-receipt profiler-gap arithmetic only; overlaps, bundled changes and independent input/GPU/calibration prevent causal attribution.', rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); r = audit(a.root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in r.items() if k not in ('rows', 'source_pins', 'literal_proof')}, indent=2))
