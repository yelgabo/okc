"""Build a deterministic private harness draft. No GPU, imports of the payload, or POST.

Source/CPU status is reported explicitly; generation alone is not readiness.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import types

import expert_top100_pair_package as pair
from expert_top100_harness_runtime import BODY
from expert_top100_pregamma import capture
from g_native_patch_ladder import flags
from prod_size_compile_g import bindings
from prod_size_qualify import helpers

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = 'dff221ae6ccc2de35d7a4c123aea3e3eb1345d3685f6eb97e52cbbfef6a0c8c4'


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def function(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, (name, len(nodes))
    return ast.get_source_segment(source, nodes[0])


def once(source, old, new):
    assert source.count(old) == 1, (old[:100], source.count(old))
    return source.replace(old, new)


def selected(source):
    """Execute actual loader/selector with an inert loader, never Torch/CUDA."""
    env = bindings(source)
    env.update(flags(source))
    calls, seen = [], {}

    class Extension:
        def okc_set_pdl(self, value):
            assert value == 0

    def loader(**kwargs):
        calls.append(kwargs)
        return Extension()

    env.update(load_inline=loader, _MA_EXTS={}, _MA_MAIN_NS=3, _MA_PDL=False,
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0))),
        _RouterExt=lambda module: module)
    for name in ('_main_defines', '_ma_ext', '_fork_native_moe', '_expert_pair_source', '_expert_pair_bind'):
        exec(function(source, name), env)
    env['_MA_ALT_SRCS']['expert_top100_pregamma'] = env['_expert_pair_source']()
    build = env['_ma_ext']

    def observe(defines=None):
        ext = build(defines)
        src = env['MOE_CUDA_SRC']
        for flag in defines:
            if flag.startswith('@'):
                src = env['_MA_ALT_SRCS'][flag[1:]]
        seen[id(ext)] = dict(flags=list(defines), sha256=sha(src))
        return ext

    env['_ma_ext'] = observe
    result = {}
    for arm in (False, True):
        state = dict(ma_sfx=('-DOKC_PRMTSC=1',), ma_sc6=True, ma_ext={})
        env['_expert_pair_bind'](state, arm)
        for t in (1, 8, 16, 32, 128, 256):
            env['_fork_native_moe'](t, state)
        for key, ext in state['ma_ext'].items():
            result[str(int(arm)) + ':' + key] = seen[id(ext)]
    return result, calls


def build():
    package, package_manifest = pair.build()
    candidate = (ROOT / 'tools/expert_top100_candidate.py').read_text(encoding='utf8')
    assert sha(candidate) == CANDIDATE, 'candidate drift'
    selection, calls = selected(package)
    original, _ = capture(pair.BASE.read_text(encoding='utf8'))
    # Binding builds B4/MAIN first. Compare by immutable module name.
    control = {c['name']: c for c in calls if c['name'] in {v['name'] for v in original}}
    assert control == {c['name']: c for c in original}, 'exact control compiler inputs'
    oracle, oracle_manifest = helpers(ROOT)
    # Reference constants may not silently change product configuration.
    def assignments(s):
        return {t.id: ast.dump(n.value) for n in ast.parse(s).body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    old, new = assignments(package), assignments(oracle)
    assert all(old[k] == new[k] for k in old.keys() & new.keys()), 'reference global collision'
    # Both arithmetic domains are finite checked. BF16 hidden and FP32 stream
    # are compared to their corresponding oracle outputs, never cross-dtype.
    oracle = once(oracle, '    if not torch.isfinite(o).all():',
        "    assert math.isfinite(float(tol)) and bool(torch.isfinite(r).all()),'EXPERT_REFERENCE_NONFINITE'\n    if not torch.isfinite(o).all():")
    owners = (ROOT / 'tools/dl_g1_layer_owners.py').read_text(encoding='utf8')
    owners = once(owners, "    arm['_wl_stage_on']=bool(on)\n", '')
    old_body = (ROOT / 'tools/dl_g1_layer_body.py').read_text(encoding='utf8')
    reused = '\n\n'.join(function(old_body, name) for name in
        ('_wl_copy', '_wl_capture', '_wl_refs', '_wl_weight_signature'))
    reused = once(reused, '    graph.replay();torch.cuda.synchronize()',
        "    data['stream'].copy_(saved[0]);data['attn_input'].copy_(saved[1])\n    graph.replay();torch.cuda.synchronize()")
    from g_common_release_budget import SUPPORT
    signature = function(SUPPORT, '_gqb_signature')
    contract = (ROOT / 'tools/expert_top100_harness_contract.py').read_text(encoding='utf8')
    core = '\nimport math,time\n' + oracle + '\n' + owners + '\n' + reused + '\n' + signature + '\n' + contract + '\n' + BODY
    pins = dict(control=pair.PIN, candidate=CANDIDATE, pair=package_manifest['source_sha256'], core=sha(core))
    source = package + '\n_EH_OFFLINE_HOLD=True\n_EH_PINS=' + repr(pins) + '\n_EH_SELECTED=' + repr(selection) + '\n' + core
    ast.parse(source)
    assert len(source.encode()) < 1_000_000, 'payload size'
    manifest = dict(source_sha256=sha(source), bytes=len(source.encode()), pins=pins, selected=selection,
        oracle=oracle_manifest, seeds=[1102031,1102037,1102049], cells=12, rows=36, placements=4,
        rounds=24, flush_bytes=256<<20, total_guard_s=295, target_fit='UNMEASURED',
        official_arm='exact607', gpu_qualified=False, dispatch_ready=False,
        status='STOP_CPU_HOST_GATE', offline_hold=True,
        remaining=['full generated prepare CPU execution with real Torch storage/version semantics',
                   'target host/device build and preparation fit',
                   'target numerical, replay, resource and performance qualification'])
    return source, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(source.encode())
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(json.dumps(manifest, indent=2))
