"""Build one private T8/T16 helper norm-aggregation comparison."""
import argparse
import ast
import json
from pathlib import Path
import subprocess

import expert_top100_moe_adapter as adapter
from cuda_issue_groups import BASE_PIN, NATIVE_PIN, ROOT, sha
from cuda_issue_groups_harness import assignment
from expert_top100_moe_harness import ADAPTER_PIN, REFERENCE_PIN, NVFP4_PIN, extract
from expert_top100_pregamma import capture
from prod_size_compile_g import bindings
from d32_b4_prefix_audit import body

HELPER_PIN = 'b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716'


def patched(helper):
    original = body(helper, '__device__ void ws_helper(')
    changed = original.replace('  for (int q = 0; q < Nh; ++q) {', '''#if OKC_WS_NORM
  float norm0 = 0.f, norm1 = 0.f;
#endif
  for (int q = 0; q < Nh; ++q) {''')
    old = '        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);'
    assert changed.count(old) == 1
    changed = changed.replace(old, '''#if OKC_WS_NORM
        if (gg == 0 && tk < cnt) {
          float& acc = h ? norm1 : norm0;
          asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(acc) : "f"(acc), "f"(sq));
          if (q + 1 == Nh || (v + 1) % RB_B == 0) {
            atomicAdd(&ss_cur()[st + tk], acc);
            acc = 0.f;
          }
        }
#else
''' + old + '\n#endif')
    assert changed != original
    return helper.replace(original, changed, 1)


def contract_source():
    source = (ROOT / 'tools/expert_top100_moe_harness_contract.py').read_text()
    changes = (
        ('row[1] >= 32 and any(', 'row[1] in (8, 16) and any('),
        ('if row[1] < 32:', 'if row[1] not in (8, 16):'),
        ('row[0] == seed and row[2] == dist]', 'row[0] == seed and row[2] == dist and row[1] in (8, 16)]'),
        (' / 6 for index', ' / 2 for index'),
    )
    for old, new in changes:
        assert source.count(old) == (2 if old == ' / 6 for index' else 1)
        source = source.replace(old, new)
    old = "    return dict(component_pass=not regressions and all(row['passed'] for row in summaries),"
    assert source.count(old) == 1
    return source.replace(old, """    unresolved_drift = [row[:3] for row in rows if row[1] not in (8, 16)
                        and any(abs(math.log(ratio)) > max(math.log(1.01), 2 * max(row[6]))
                                for ratio in row[5])]
    return dict(component_pass=not regressions and not unresolved_drift
                and all(row['passed'] for row in summaries), unresolved_drift=unresolved_drift,""")


def build():
    prefix, _ = adapter.build()
    assert sha(prefix) == ADAPTER_PIN
    helper = bindings(prefix)['_MA_ALT_SRCS']['moe_helper0']
    assert sha(helper) == HELPER_PIN and prefix.count(helper) == 1
    modified = patched(helper)
    prefix = prefix.replace(helper, modified, 1)
    old = """    if arm and key in ('b4', 'main'):
        defines = tuple(defines) + ('@expert_top100_pregamma',)"""
    new = """    if arm and key in ('tiny2', 'mid'):
        defines = tuple(defines) + ('-DOKC_WS_NORM=1',)"""
    assert prefix.count(old) == 1
    prefix = prefix.replace(old, new)
    prefix = assignment(prefix, '_EXPERT_PAIR_EDITS', ())
    prefix = assignment(prefix, 'EXPERT_MOE_IDENTITIES',
                        ('gopt_ws_norm', BASE_PIN, HELPER_PIN, sha(modified)))
    controls, routes = capture(prefix)
    candidates, candidate_routes = capture(prefix.replace("arm = state.get('_expert_pair_arm', False)", 'arm = True'))
    assert routes == candidate_routes and len(controls) == len(candidates) == 5
    selection = []
    for route, control, candidate in zip(routes[:5], controls, candidates):
        affected = route['key'] in ('tiny2', 'mid')
        assert control['cuda_sources'] == candidate['cuda_sources']
        assert control['cpp_sources'] == candidate['cpp_sources']
        assert [f for f in candidate['extra_cuda_cflags'] if f != '-DOKC_WS_NORM=1'] == control['extra_cuda_cflags']
        assert ('-DOKC_WS_NORM=1' in candidate['extra_cuda_cflags']) == affected
        if route['key'] in ('b4', 'main'):
            assert sha(control['cuda_sources']) == NATIVE_PIN
        selection.append(dict(key=route['key'], affected=affected, cuda_sha256=sha(control['cuda_sources']),
                              control_flags=control['extra_cuda_cflags'], candidate_flags=candidate['extra_cuda_cflags']))
    unique = controls + [call for call in candidates if call not in controls]
    assert len(unique) == 7
    descriptors = [{**call, **{k: sha(call[k]) for k in ('cpp_sources', 'cuda_sources')}} for call in unique]
    reference = (ROOT / 'muse_moe_w4a16/reference.py').read_text()
    nvfp4 = (ROOT / 'muse_moe_w4a16/nvfp4.py').read_text()
    assert sha(reference) == REFERENCE_PIN and sha(nvfp4) == NVFP4_PIN
    route_source = '\n\n'.join((
        'D_HIDDEN = D_EXPERT = 4096\nN_EXPERTS = 256\nROUTED_PER_TOKEN = 8\nPOPULARITY = .03\nLOCAL = {"ep8": (0, 32), "full": (0, 256)}',
        extract(nvfp4, 'outlier_channels'), extract(reference, 'generate_input')))
    contract = contract_source()
    runtime = runtime_source()
    identity = sha(prefix + route_source + contract + runtime + json.dumps(descriptors, sort_keys=True))
    source = prefix + '\n' + route_source + '\n' + contract + '\n'
    source += f'H_RUNTIME_ID = {identity!r}\nH_HOLD = False\nH_BUILD_CALLS = {descriptors!r}\n' + runtime
    ast.parse(source)
    assert len(source.encode()) < 1_000_000
    manifest = dict(experiment='GOPT-WS-03', source_sha256=sha(source), bytes=len(source.encode()),
                    runtime_id=identity, base_sha256=BASE_PIN, helper_control_sha256=HELPER_PIN,
                    helper_edited_sha256=sha(modified), native_sha256=NATIVE_PIN,
                    official_arm='candidate2', hold=False, compiler_calls=descriptors, selection=selection,
                    contract_sha256=sha(contract), runtime_sha256=sha(runtime), routes=routes,
                    public_submission=False, target_qualified=False,
                    gate='T8/T16 all four affected crosses >=1% and > twice duplicate null in every bank/distribution; no affected >=1% regression; unchanged sizes within max(1%, twice row null)')
    return source, manifest, controls[1:3], candidates[1:3]


def runtime_source():
    source = (ROOT / 'tools/expert_top100_moe_harness_runtime.py').read_text()
    old = '    report = None\n    try:'
    new = '''    report = None

    def progress(stage, error=None):
        message = dict(runtime_id=H_RUNTIME_ID, stage=stage, elapsed_s=time.monotonic()-began,
                       completed_rows=len(rows), last_row=rows[-1] if rows else None,
                       counters=counters, error=error)
        encoded = json.dumps(message, separators=(',', ':'), allow_nan=False)
        require(len(encoded.encode()) < 1600, 'bounded progress')
        print('GOPT_WS_NORM_PROGRESS ' + encoded, flush=True)

    try:'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = '        owners = _h_ownership(states)'
    source = source.replace(old, old + "\n        progress('owners_validated')", 1)
    old = '                        rows.append(row)'
    source = source.replace(old, old + "\n                        progress('row_complete')", 1)
    old = '    finally:\n        load_inline = original_loader'
    new = '''    except BaseException as error:
        try:
            progress('failed', str(error)[:256])
        except Exception:
            print('GOPT_WS_NORM_PROGRESS serialization_failed', flush=True)
        raise
    finally:
        load_inline = original_loader'''
    assert source.count(old) == 1
    return source.replace(old, new)


def compile_pairs(out, controls, candidates):
    from gopt_skinny_depth import compile_pairs as compile_depth
    import inspect
    source = inspect.getsource(compile_depth)
    source = source.replace("('tiny', 'tiny2', 'mid')", "('tiny2', 'mid')")
    namespace = dict(json=json, subprocess=subprocess, sha=sha)
    exec(source, namespace)
    namespace['compile_pairs'](out, controls, candidates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compile-out', type=Path)
    args = parser.parse_args()
    source, manifest, controls, candidates = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('experiment', 'source_sha256', 'bytes', 'runtime_id')}), flush=True)
    if args.compile_out:
        compile_pairs(args.compile_out, controls, candidates)
