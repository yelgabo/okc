"""Build the next Family G file, optionally with the off-board empty-routing replay check."""
import argparse
import ast
import hashlib
import inspect
from pathlib import Path

from dl_fork_marlin_fallback import patch_fallback
from dl_fork_moe_empty import patch_empty_routing
from dl_fork_moe_empty_probe import empty_conformance

BASE_SHA = '258c848c662149f9c6ac178039f179de6f0b0ec4cb0b503451d2974c7a7a8dfc'
ATT7_SHA = '0754d5ffd18e00614787c6bbb8a78c59f503fb9ccc2691bd10420fb65f7729d7'
PROBE_WRAPPER = '''
_g1b3_release_prepare = prepare
def prepare(weights):
    state = _g1b3_release_prepare(weights)
    if not state['_fork_native_eligible']:
        raise RuntimeError('Native empty-routing check requires a selected native backend')
    control = {}
    _fork_prepare_marlin(weights, control)
    empty_conformance(torch, state, _moe, _fork_native_moe,
                      lambda m, st: control['_fork_marlin_adapter'])
    return state
'''


def round_native_attention(source):
    """Keep the BF16 value-vector boundary used by both official norm/gate calls."""
    node = next(
        n for n in ast.parse(source).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'ATT7_CUDA_SRC'
    )
    device = ast.literal_eval(node.value)
    if hashlib.sha256(device.encode()).hexdigest() != ATT7_SHA:
        raise ValueError('unreviewed native attention source')
    old = '__device__ __forceinline__ float pr8(float x) {\n  return x;\n}'
    new = old.replace('return x;', 'return __bfloat162float(__float2bfloat16_rn(x));')
    if device.count(old) != 1 or device.count('pr8(') != 16 or source.count(old) != 1:
        raise ValueError('unexpected native output conversion sites')
    return source.replace(old, new, 1)


def build(source, probe=False, bf16_attention=False):
    if hashlib.sha256(source.encode()).hexdigest() != BASE_SHA:
        raise ValueError('expected the exact G1b2 H200 candidate')
    source = patch_fallback(patch_empty_routing(source))
    source = source.replace('# Modified by yelgabo: G1b2,', '# Modified by yelgabo: G1b3, empty routing and Marlin fallback,')
    source = source.replace('"G1b2 backend"', '"G1b3 backend"')
    if bf16_attention:
        source = round_native_attention(source)
        source = source.replace('# Modified by yelgabo: G1b3,', '# Modified by yelgabo: G1b4, BF16 attention value,')
        source = source.replace('"G1b3 backend"', '"G1b4 backend"')
    if probe:
        source += '\n# CHECK ONLY: additional conformance work with a private Marlin control.\n'
        source += inspect.getsource(empty_conformance) + '\n' + PROBE_WRAPPER
    compile(source, '<Family G qualification>', 'exec')
    payload = source.encode()
    if len(payload) >= 1_000_000:
        raise ValueError(f'candidate too large: {len(payload)}')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--probe', action='store_true', help='include off-board eager and graph checks')
    parser.add_argument('--bf16-attention', action='store_true', help='restore the native attention value round before RMS/gate (G1b4)')
    args = parser.parse_args()
    payload = build(args.input.read_text('utf-8'), args.probe, args.bf16_attention)
    args.output.write_bytes(payload)
    print(len(payload), hashlib.sha256(payload).hexdigest())
