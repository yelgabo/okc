"""Generate the fixed two-launch WS complete-call comparison."""
import argparse
import ast
import json
from pathlib import Path

import expert_top100_moe_adapter as adapter
from cuda_issue_groups import BASE_PIN, NATIVE_PIN, ROOT, sha
from cuda_issue_groups_harness import assignment
from expert_top100_moe_harness import ADAPTER_PIN, REFERENCE_PIN, NVFP4_PIN, extract
from expert_top100_pregamma import capture
from gopt_ws_norm import HELPER_PIN
from prod_size_compile_g import bindings


FINALIZER = r'''
#if OKC_WS_SPLIT
__global__ void __launch_bounds__(256) ws_finalize(Params p) {
  const int tid = threadIdx.x, it = blockIdx.x;
  constexpr int FSEG = HD / (256 * 4);
  const int tk = it / FSEG, col = (it % FSEG) * (256 * 4) + tid * 4;
  const int gen = p.ws_meta[64];
  const float* ss = (gen & 1) ? p.ss1 : p.ss;
  int qvp[8];
  float wvp[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    qvp[s] = (int)reinterpret_cast<const short*>(p.ws_meta)[tk * 8 + s];
    wvp[s] = qvp[s] >= 0 ? __ldg(p.topk_w + tk * 8 + s) : 0.f;
  }
  float ssv[8];
  float4 yv[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    ssv[s] = qvp[s] >= 0 ? __ldcg(ss + qvp[s]) : 0.f;
    yv[s] = qvp[s] >= 0 ? __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qvp[s] * HD + col)) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    if (qvp[s] >= 0) {
      const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);
      acc.x += r * yv[s].x; acc.y += r * yv[s].y; acc.z += r * yv[s].z; acc.w += r * yv[s].w;
    }
  }
  __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
  uint2 o;
  o.x = *reinterpret_cast<uint32_t*>(&o0);
  o.y = *reinterpret_cast<uint32_t*>(&o1);
  *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;
  const int i = it * 256 + tid;
  if (i < JMAX) p.doneA[i] = 0;
  if (i < 2) p.ctr[i] = 0;
  if (i == 0) p.ctr[3] = (int)((unsigned)gen + 1u);
}
#endif
'''


def transform(helper):
    source = helper
    def once(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    source = '#ifndef OKC_WS_SPLIT\n#define OKC_WS_SPLIT 0\n#endif\n' + source
    once('struct Params {', 'struct Params {\n#if OKC_WS_SPLIT\n  int* ws_meta;\n#endif')
    anchor = '#if OKC_WS\n  WS_JOIN();  // the helper\'s last epilogue stores are visible\n#endif'
    once(anchor, anchor + '''
#if OKC_WS_SPLIT
  static_assert(OKC_WS && !OKC_WGS && NT == 256, "WS split selection");
  if (blockIdx.x == 0) {
    for (int i = tid; i < p.M * 8; i += NT)
      reinterpret_cast<short*>(p.ws_meta)[i] = S.pos[i];
    if (tid == 0) p.ws_meta[64] = S.gen;
  }
  return;
#endif''')
    once('}  // namespace okc', FINALIZER + '\n}  // namespace okc')
    anchor = '  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;'
    once(anchor, '''#if OKC_WS_SPLIT
  TORCH_CHECK(scratch.size() == 4 && scratch[3].scalar_type() == torch::kInt32
              && scratch[3].numel() == 128, "WS_SPLIT metadata ABI");
  TORCH_CHECK(p.M <= 16, "WS_SPLIT token bound");
  p.ws_meta = (int*)scratch[3].data_ptr();
  p.dbg = nullptr;
#else
''' + anchor + '\n#endif')
    anchor = '  TORCH_CHECK(launch==cudaSuccess,"FORK_MOE_LAUNCH ",cudaGetErrorString(launch));'
    once(anchor, anchor + '''
#if OKC_WS_SPLIT
  okc::ws_finalize<<<p.M * 4, 256, 0, st>>>(p);
  const cudaError_t finish = cudaGetLastError();
  TORCH_CHECK(finish == cudaSuccess, "WS_SPLIT_FINALIZER ", cudaGetErrorString(finish));
#endif''')
    source += r'''
void okc_ws_finalize_probe(std::vector<torch::Tensor> a) {
#if OKC_WS_SPLIT
  TORCH_CHECK(a.size() == 7 && a[4].size(0) > 0 && a[4].size(0) <= 16,
              "WS_SPLIT probe ABI");
  okc::Params p{};
  p.M = (int)a[4].size(0);
  p.ws_meta = (int*)a[0].data_ptr();
  p.ss = (float*)a[1].data_ptr(); p.ss1 = p.ss + okc::EMAX;
  p.ybuf = (float*)a[2].data_ptr(); p.topk_w = (const float*)a[3].data_ptr();
  p.out = (__nv_bfloat16*)a[4].data_ptr();
  p.ctr = (int*)a[5].data_ptr(); p.doneA = (int*)a[6].data_ptr();
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  okc::ws_finalize<<<p.M * 4, 256, 0, st>>>(p);
  const cudaError_t result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "WS_SPLIT probe launch ", cudaGetErrorString(result));
#else
  TORCH_CHECK(false, "WS_SPLIT probe requires split build");
#endif
}
'''
    return source


SCRATCH = '''
_WS_SCRATCH_BASE = _ma_make_scratch
def _ma_make_scratch(device):
    return _WS_SCRATCH_BASE(device) + [torch.zeros(128, dtype=torch.int32, device=device)]
'''


def contract_source():
    source = (ROOT / 'tools/expert_top100_moe_harness_contract.py').read_text()
    source = source.replace('if row[1] >= 32 and any(', 'if row[1] < 32 and any(')
    source = source.replace('if row[1] < 32:\n            drift.append', 'if row[1] >= 32:\n            drift.append')
    old = "    return dict(component_pass=not regressions and all(row['passed'] for row in summaries),"
    assert source.count(old) == 1
    return source.replace(old, """    unresolved_drift = [row[:3] for row in rows if row[1] >= 32
                        and any(abs(math.log(ratio)) > max(math.log(1.01), 2 * max(row[6]))
                                for ratio in row[5])]
    return dict(component_pass=not regressions and not unresolved_drift
                and all(row['passed'] for row in summaries), unresolved_drift=unresolved_drift,""")


def runtime_source():
    source = (ROOT / 'tools/expert_top100_moe_harness_runtime.py').read_text()
    def once(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    once('def prepare(weights):', 'def _ws_prepare_inner(weights):')
    once('        owners = _h_ownership(states)', '        owners = _h_ownership(states)\n        _ws_probe(states, deadline, counters)')
    once('    report = None\n', '    report = None\n    _WS_LIVE.update(rows=rows, counters=counters, began=began, stage="preparation")\n')
    once('                    row = _h_case(', '                    _WS_LIVE["stage"] = [seed, tokens, dist]\n                    row = _h_case(')
    once('            for state in states:\n                _expert_moe_check_guards(state)',
         '            _ws_validate_metadata(states, data)\n            for state in states:\n                _expert_moe_check_guards(state)')
    once('    finally:\n        load_inline = original_loader\n        _h_reset(states)', '''    except BaseException as error:
        _WS_LIVE['primary_error'] = error
        raise
    finally:
        load_inline = original_loader
        _WS_LIVE['stage'] = 'cleanup'
        try:
            _h_reset(states)
        except BaseException as cleanup_error:
            _WS_LIVE['cleanup_error'] = str(cleanup_error)[:300]
            if 'primary_error' in _WS_LIVE:
                raise _WS_LIVE['primary_error'] from cleanup_error
            raise''')
    once('    _expert_moe_budget(deadline)\n    report.update(', '    _WS_LIVE["stage"] = "post-cleanup-budget"\n    _expert_moe_budget(deadline)\n    report.update(')
    once("    encoded = json.dumps(report, separators=(',', ':'), allow_nan=False)",
         '    _WS_LIVE["stage"] = "serialization"\n    encoded = _ws_encode(report)')
    once("    require(len(encoded.encode()) < CAPSULE_LIMIT, 'bounded capsule')", "    require(len(encoded) < 12000, 'bounded capsule')\n    _WS_LIVE['stage'] = 'final-budget'")
    once("print('EXPERT_MOE_CAPSULE ' + encoded, flush=True)", "print('GOPT_WS_SPLIT_REPORT ' + encoded, flush=True)")
    return (ROOT / 'tools/gopt_ws_split_runtime.py').read_text() + '\n' + source + '''

def prepare(weights):
    _WS_LIVE.clear()
    _WS_LIVE.update(rows=[], counters={}, stage='entry', began=time.monotonic())
    try:
        return _ws_prepare_inner(weights)
    except BaseException as error:
        try:
            _ws_partial(error)
        except BaseException as report_error:
            print('GOPT_WS_SPLIT_REPORTING_FAILURE ' + str(report_error)[:300], flush=True)
        raise
'''


def build():
    prefix, _ = adapter.build()
    assert sha(prefix) == ADAPTER_PIN
    helper = bindings(prefix)['_MA_ALT_SRCS']['moe_helper0']
    assert sha(helper) == HELPER_PIN
    modified = transform(helper)
    prefix = prefix.replace(helper, modified, 1)
    old = """    if arm and key in ('b4', 'main'):
        defines = tuple(defines) + ('@expert_top100_pregamma',)"""
    new = """    if arm and key in ('tiny', 'tiny2', 'mid'):
        defines = tuple(defines) + ('-DOKC_WS_SPLIT=1',)"""
    assert prefix.count(old) == 1
    prefix = prefix.replace(old, new)
    prefix = assignment(prefix, '_EXPERT_PAIR_EDITS', ())
    prefix = assignment(prefix, 'EXPERT_MOE_IDENTITIES', ('gopt_ws_split', BASE_PIN, HELPER_PIN, sha(modified)))
    old = 'cpp_sources=MOE_CPP_SRC,'
    new = 'cpp_sources=MOE_CPP_SRC + ("\\nvoid okc_ws_finalize_probe(std::vector<torch::Tensor>);" if "okc_ws_finalize_probe" in src else ""),'
    assert prefix.count(old) == 1
    prefix = prefix.replace(old, new)
    old = 'functions=["okc_run", "okc_set_pdl"] +'
    new = 'functions=["okc_run", "okc_set_pdl"] + (["okc_ws_finalize_probe"] if "okc_ws_finalize_probe" in src else []) +'
    assert prefix.count(old) == 1
    prefix = prefix.replace(old, new)
    prefix += '\n' + SCRATCH
    controls, routes = capture(prefix)
    candidates, other_routes = capture(prefix.replace("arm = state.get('_expert_pair_arm', False)", 'arm = True'))
    assert routes == other_routes
    descriptors, selection = [], []
    for route, control, candidate in zip(routes[:5], controls, candidates):
        affected = route['key'] in ('tiny', 'tiny2', 'mid')
        assert control['cuda_sources'] == candidate['cuda_sources']
        assert [x for x in candidate['extra_cuda_cflags'] if x != '-DOKC_WS_SPLIT=1'] == control['extra_cuda_cflags']
        assert ('-DOKC_WS_SPLIT=1' in candidate['extra_cuda_cflags']) == affected
        if not affected:
            assert control == candidate and sha(control['cuda_sources']) == NATIVE_PIN
        selection.append(dict(key=route['key'], affected=affected, control_flags=control['extra_cuda_cflags'], candidate_flags=candidate['extra_cuda_cflags']))
    unique = controls + [call for call in candidates if call not in controls]
    assert len(unique) == 8
    for call in unique:
        descriptors.append({**call, **{key: sha(call[key]) for key in ('cpp_sources', 'cuda_sources')}})
    reference = (ROOT / 'muse_moe_w4a16/reference.py').read_text()
    nvfp4 = (ROOT / 'muse_moe_w4a16/nvfp4.py').read_text()
    assert sha(reference) == REFERENCE_PIN and sha(nvfp4) == NVFP4_PIN
    route_source = '\n\n'.join(('D_HIDDEN = D_EXPERT = 4096\nN_EXPERTS = 256\nROUTED_PER_TOKEN = 8\nPOPULARITY = .03\nLOCAL = {"ep8": (0, 32), "full": (0, 256)}', extract(nvfp4, 'outlier_channels'), extract(reference, 'generate_input')))
    contract, runtime = contract_source(), runtime_source()
    identity = sha(prefix + route_source + contract + runtime + json.dumps(descriptors, sort_keys=True))
    source = prefix + '\n' + route_source + '\n' + contract + f'\nH_RUNTIME_ID={identity!r}\nH_HOLD=False\nH_BUILD_CALLS={descriptors!r}\n' + runtime
    ast.parse(source)
    manifest = dict(experiment='GOPT-WS-SPLIT-04', source_sha256=sha(source), bytes=len(source.encode()), runtime_id=identity,
                    base_sha256=BASE_PIN, helper_control_sha256=HELPER_PIN, helper_edited_sha256=sha(modified), native_sha256=NATIVE_PIN,
                    official_arm='candidate2', hold=False, compiler_calls=descriptors, selection=selection, routes=routes,
                    contract_sha256=sha(contract), runtime_sha256=sha(runtime), public_submission=False,
                    gate='All six complete-call geomean>=1.01 and >twice duplicate-log-null in all four crosses, each bank/distribution; no affected1% regression; unchanged drift <=max1%/twice-null; complete correctness/coverage')
    return source, manifest, controls[:3], candidates[:3]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, manifest, _, _ = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: manifest[key] for key in ('experiment', 'source_sha256', 'bytes', 'runtime_id')}))
