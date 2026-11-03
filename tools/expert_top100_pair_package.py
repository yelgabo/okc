"""Offline dual-build packaging only; no timing harness or dispatch permission."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from expert_top100_pregamma import BASE, PIN, transform
from prod_size_compile_g import bindings

SUPPORT = '''
# PRIVATE dual-build preparation support. Never submit this diagnostic publicly.
_EXPERT_PAIR_EDITS = {edits!r}
def _expert_pair_source():
    source = MOE_CUDA_SRC
    for old, new, count in _EXPERT_PAIR_EDITS:
        if source.count(old) != count:
            raise RuntimeError('EXPERT_PAIR_SOURCE_DRIFT')
        source = source.replace(old, new)
    return source
_MA_ALT_SRCS['expert_top100_pregamma'] = _expert_pair_source()

def _expert_pair_bind(state, arm):
    # Call only on a separately owned prepared state, before any captures.
    if type(arm) is not bool or '_expert_pair_arm_bound' in state:
        raise RuntimeError('EXPERT_PAIR_INVALID_REBIND')
    state['_expert_pair_arm'] = arm
    state['_expert_pair_arm_bound'] = arm
    for key in ('b4', 'main'):
        state['ma_ext'].pop(key, None)
    for tokens in (32, 128):
        _fork_native_moe(tokens, state)
    return state
'''


def build():
    source = BASE.read_text(encoding='utf8')
    assert hashlib.sha256(source.encode()).hexdigest() == PIN
    native = bindings(source)['MOE_CUDA_SRC']
    candidate, edits = transform(native)
    anchor = '    ext = state["ma_ext"].get(key)'
    assert source.count(anchor) == 1
    dispatch = '''    arm = state.get('_expert_pair_arm', False)
    if type(arm) is not bool or state.get('_expert_pair_arm_bound', arm) is not arm:
        raise RuntimeError('EXPERT_PAIR_ARM_MUTATED')
    if arm and key in ('b4', 'main'):
        defines = tuple(defines) + ('@expert_top100_pregamma',)
'''
    source = source.replace(anchor, dispatch + anchor)
    source += SUPPORT.format(edits=edits)
    ast.parse(source)
    assert len(source.encode()) < 1_000_000
    return source, dict(base_sha256=PIN,
        native_control_sha256=hashlib.sha256(native.encode()).hexdigest(),
        native_candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        source_sha256=hashlib.sha256(source.encode()).hexdigest(), bytes=len(source.encode()),
        default_arm='control', scope='dual-build source packaging; not a complete measurement harness',
        pending=['independent mutable preparations and ownership validation',
                 'candidate replay correctness', 'matched timing and strict owning reader',
                 'target compile and preparation budget'], gpu_qualified=False, dispatch_ready=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(source.encode())
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))
