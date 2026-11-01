"""Build the next Family G file, optionally with the off-board empty-routing replay check."""
import argparse
import hashlib
import inspect
from pathlib import Path

from dl_fork_marlin_fallback import patch_fallback
from dl_fork_moe_empty import patch_empty_routing
from dl_fork_moe_empty_probe import empty_conformance

BASE_SHA = '258c848c662149f9c6ac178039f179de6f0b0ec4cb0b503451d2974c7a7a8dfc'
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


def build(source, probe=False):
    if hashlib.sha256(source.encode()).hexdigest() != BASE_SHA:
        raise ValueError('expected the exact G1b2 H200 candidate')
    source = patch_fallback(patch_empty_routing(source))
    source = source.replace('# Modified by yelgabo: G1b2,', '# Modified by yelgabo: G1b3, empty routing and Marlin fallback,')
    source = source.replace('"G1b2 backend"', '"G1b3 backend"')
    if probe:
        source += '\n# CHECK ONLY: additional conformance work with a private Marlin control.\n'
        source += inspect.getsource(empty_conformance) + '\n' + PROBE_WRAPPER
    compile(source, '<G1b3>', 'exec')
    payload = source.encode()
    if len(payload) >= 1_000_000:
        raise ValueError(f'candidate too large: {len(payload)}')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--probe', action='store_true', help='include off-board eager and graph checks')
    args = parser.parse_args()
    payload = build(args.input.read_text('utf-8'), args.probe)
    args.output.write_bytes(payload)
    print(len(payload), hashlib.sha256(payload).hexdigest())
