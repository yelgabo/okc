"""Build the fixed T8/T16 full-K row-owned private comparison."""
import argparse
import ast
import json
from pathlib import Path

from cuda_issue_groups import ROOT, sha
from cuda_issue_groups_harness import assignment
from expert_top100_moe_harness import extract, REFERENCE_PIN, NVFP4_PIN
from expert_top100_pregamma import capture
from prod_size_compile_g import bindings
from gopt_ws_norm import HELPER_PIN
from gopt_ws_split import runtime_source as split_runtime

RELEASE_PIN = '5df257afd6ae2863bc706826a9798d97b17e73830f1d10703c81bba21203eb9e'


def native_source(helper):
    assert sha(helper) == HELPER_PIN
    prelude = helper[:helper.index('// ---------------------------------------------------------------- skinny path:')]
    left = prelude.index('#if !OKC_WGS\n  float red[')
    right = prelude.index('#endif', left) + len('#endif')
    prelude = prelude[:left] + prelude[right:]
    scales = helper[helper.index('#if OKC_PRMTSC\n// e4m3'):helper.index('__device__ __forceinline__ void mbar_init')]
    final = helper[helper.index('#if OKC_FINPF && OKC_FINCOL && !OKC_WGS', helper.index('__global__ void __launch_bounds__')):helper.index('\n}  // namespace okc')]
    source = prelude + scales + (ROOT / 'tools/gopt_ws_owned_kernel.cu').read_text() + '\n' + final
    host = helper[helper.index('\n}  // namespace okc'):]
    left = host.index('static size_t okc_smem_bytes()')
    right = host.index('\nstatic int g_okc_pdl', left)
    host = host[:left] + '''static size_t okc_smem_bytes() {
  return ((sizeof(okc::Smem) + 127) / 128) * 128 + 3 * 36864 + 8 * 1032 * 2;
}
''' + host[right:]
    host = host.replace('p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();',
                        'TORCH_CHECK(w.size() == 16, "OWN packed ABI");\n  p.wAu = (const unsigned char*)w[14].data_ptr(); p.wBu = (const unsigned char*)w[15].data_ptr();')
    host = host.replace('  size_t sm = okc_smem_bytes();', '''  TORCH_CHECK((p.M == 8 || p.M == 16) && grid == 132, "OWN shape/grid");
  size_t sm = okc_smem_bytes();
  TORCH_CHECK(sm <= 160 * 1024, "OWN shared cap");''')
    return source + host


PACKING = '''
def _owned_pack(original):
    raw = original.view(torch.uint8).reshape(-1, 4, 36864)
    result = torch.empty_like(raw).reshape(-1, 4, 36864)
    for stage in range(4):
        for warp in range(4):
            for half in range(2):
                k = 2 * stage + half
                result[:, stage, warp*8192+half*4096:warp*8192+(half+1)*4096].copy_(raw[:, warp, k*4096:(k+1)*4096])
                result[:, stage, 32768+warp*1024+half*512:32768+warp*1024+(half+1)*512].copy_(raw[:, warp, 32768+k*512:32768+(k+1)*512])
    for stage in range(4):
        for warp in range(4):
            for half in range(2):
                k = 2 * stage + half
                require(torch.equal(result[:, stage, warp*8192+half*4096:warp*8192+(half+1)*4096], raw[:, warp, k*4096:(k+1)*4096]), 'OWN code round trip')
                require(torch.equal(result[:, stage, 32768+warp*1024+half*512:32768+warp*1024+(half+1)*512], raw[:, warp, 32768+k*512:32768+(k+1)*512]), 'OWN scale round trip')
    return result.reshape(-1)
'''

FP32_PROBE = '''
def _owned_fp32_probes(weights, states, deadline, counters):
    def decode(packed, scales, gs):
        p = packed.view(torch.uint8)
        codes = torch.stack((p & 15, (p >> 4) & 15), -1).flatten(-2)
        grid = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), dtype=torch.float32, device=p.device)
        values = grid[(codes & 7).long()]
        values = torch.where((codes & 8) != 0, -values, values)
        return (values.reshape(-1, 256, 16) * (scales.float() * gs).unsqueeze(-1)).reshape(values.shape)
    device = states[0]['device']
    saved_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        a = decode(weights['w13'][0], weights['w13_sf'][0], weights['w13_gs'][0])
        b = decode(weights['w2'][0], weights['w2_sf'][0], weights['w2_gs'][0])
        for tokens in (8, 16):
            _expert_moe_budget(deadline)
            data = generate_input(tokens, 'ep8', 19137 + tokens, device=device)
            ids = torch.tensor((0,32,33,34,35,36,37,38), device=device, dtype=torch.int32)
            data['topk_ids'].copy_(ids[None,:].expand(tokens,8))
            h = data['hidden_states'].float() @ a.T
            inter = torch.nn.functional.silu(h[:,:4096]) * h[:,4096:]
            y = inter @ b.T
            y = y * (1. + weights['post_norm_w'].float())
            expected = (y * torch.rsqrt((y*y).mean(-1,keepdim=True) + 1e-8) * data['topk_weights'][:,:1]).to(torch.bfloat16)
            for state in states:
                output = _expert_moe_run(data,state,state['_expert_pair_arm'])
                errors = _h_errors(output,expected)
                counters['worst'] = [max(x,y) for x,y in zip(counters['worst'],errors)]
                counters['fp32_reference_cases'] = counters.get('fp32_reference_cases',0) + 1
        torch.cuda.synchronize(device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved_tf32
'''


def contract_source():
    source = (ROOT / 'tools/expert_top100_moe_harness_contract.py').read_text()
    source = source.replace('row[1] >= 32 and any(', 'row[1] in (8, 16) and any(')
    source = source.replace('if row[1] < 32:', 'if row[1] not in (8, 16):')
    old = "    return dict(component_pass=not regressions and all(row['passed'] for row in summaries),"
    assert source.count(old) == 1
    return source.replace(old, """    unresolved_drift = [row[:3] for row in rows if row[1] not in (8, 16)
                        and any(abs(math.log(ratio)) > max(math.log(1.01), 2 * max(row[6]))
                                for ratio in row[5])]
    return dict(component_pass=not regressions and not unresolved_drift
                and all(row['passed'] for row in summaries), unresolved_drift=unresolved_drift,""")


def runtime_source():
    source = split_runtime()
    source = source.replace('        _ws_probe(states, deadline, counters)\n', '')
    source = source.replace('            _ws_validate_metadata(states, data)\n', '')
    source = source.replace('        owners = _h_ownership(states)',
                            '        owners = _h_ownership(states)\n        _owned_fp32_probes(weights, states, deadline, counters)')
    start = source.index('def _ws_probe(')
    end = source.index('"""Appended unchanged', start)
    source = source[:start] + source[end:]
    start = source.index('def _ws_validate_metadata(')
    end = source.index('def _ws_encode(', start)
    source = source[:start] + source[end:]
    source = source.replace('                        rows.append(row)', '''                        rows.append(row)
                        progress = dict(runtime_id=H_RUNTIME_ID, stage='row_complete', rows=rows, counters=counters)
                        blob = _ws_encode(progress)
                        require(len(blob) < 12000, 'bounded cumulative progress')
                        print('GOPT_WS_OWN_PROGRESS ' + blob, flush=True)''')
    return FP32_PROBE + source.replace('GOPT_WS_SPLIT', 'GOPT_WS_OWN')


def build():
    release = (ROOT / 'muse_moe_w4a16/cand/current_control_release.py').read_text()
    assert sha(release) == RELEASE_PIN
    env = bindings(release)
    helper = env['_MA_ALT_SRCS']['moe_helper0']
    native = native_source(helper)
    prefix = assignment(release, 'EXPERT_MOE_IDENTITIES', ('gopt_ws_owned', RELEASE_PIN, HELPER_PIN, sha(native)))
    prefix += '\n_MA_ALT_SRCS["gopt_ws_owned"] = ' + repr(native) + '\n' + PACKING
    old = "    ext = state[\"ma_ext\"].get(key)"
    assert prefix.count(old) == 1
    prefix = prefix.replace(old, '''    if arm and M in (8, 16):
        key = 'owned'
        defines = ('-DOKC_NW=4', '-DOKC_TINY=1', '-DOKC_WS=1', '-DOKC_WGS=0',
                   '-DOKC_FAST1=0', '-DOKC_SPLIT=1', '-DOKC_FINPF=1', '@gopt_ws_owned')
''' + old)
    prefix = prefix.replace("if arm is not False or '_expert_pair_arm_bound' in state:", "if type(arm) is not bool or '_expert_pair_arm_bound' in state:")
    prefix = prefix.replace('        _fork_native_ma_prepare(local, state)', '''        _fork_native_ma_prepare(local, state)
        if arm:
            for index in (10, 11):
                state['ma_w'].append(_owned_pack(state['ma_w'][index]))
                _expert_moe_budget(deadline)''')
    prefix = prefix.replace("        if set(state['ma_ext']) != expected:", "        if set(state['ma_ext']) != (expected - {'tiny2', 'mid'} | {'owned'} if arm else expected):")
    baseline_calls, baseline_routes = capture(release)
    controls, routes = capture(prefix)
    candidates, candidate_routes = capture(prefix.replace("arm = state.get('_expert_pair_arm', False)", 'arm = True'))
    assert baseline_calls == controls and routes == baseline_routes
    unique = controls + [c for c in candidates if c not in controls]
    assert len(unique) == 6
    assert candidate_routes == [dict(tokens=r['tokens'], key='owned' if r['tokens'] in (8,16) else r['key']) for r in routes]
    descriptors = [{**c, **{k: sha(c[k]) for k in ('cpp_sources', 'cuda_sources')}} for c in unique]
    reference = (ROOT / 'muse_moe_w4a16/reference.py').read_text()
    nvfp4 = (ROOT / 'muse_moe_w4a16/nvfp4.py').read_text()
    assert sha(reference) == REFERENCE_PIN and sha(nvfp4) == NVFP4_PIN
    route_source = '\n\n'.join(('D_HIDDEN = D_EXPERT = 4096\nN_EXPERTS = 256\nROUTED_PER_TOKEN = 8\nPOPULARITY = .03\nLOCAL = {"ep8": (0, 32), "full": (0, 256)}', extract(nvfp4, 'outlier_channels'), extract(reference, 'generate_input')))
    contract, runtime = contract_source(), runtime_source()
    identity = sha(prefix + route_source + contract + runtime + json.dumps(descriptors, sort_keys=True))
    source = prefix + '\n' + route_source + '\n' + contract + f'\nH_RUNTIME_ID={identity!r}\nH_HOLD=False\nH_BUILD_CALLS={descriptors!r}\n' + runtime
    ast.parse(source)
    manifest = dict(experiment='GOPT-WS-OWN-05', source_sha256=sha(source), bytes=len(source.encode()), runtime_id=identity,
                    release_sha256=RELEASE_PIN, helper_control_sha256=HELPER_PIN, owned_native_sha256=sha(native),
                    compiler_calls=descriptors, baseline_routes=routes, candidate_routes=candidate_routes,
                    official_arm='candidate2', contract_sha256=sha(contract), runtime_sha256=sha(runtime),
                    public_submission=False, target_qualified=False, hold=False,
                    gate='All six complete-call geomean>=1.01 and >twice duplicate-log-null in every four-cross bank/distribution; no affected T8/T16 1% regression; unchanged drift<=max1%/twice-null; all correctness/coverage')
    return source, manifest, unique[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, manifest, _ = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:manifest[k] for k in ('experiment','source_sha256','bytes','owned_native_sha256')}))
