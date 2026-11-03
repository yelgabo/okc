"""Package the frozen queue treatment without the private timing comparison."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from moe_bqueue_harness import build as paired
from prod_size_compile_g import bindings


def build():
    source, manifest = paired()
    marker = '\nD_HIDDEN = D_EXPERT = 4096\n'
    assert source.count(marker) == 1
    prefix = source.split(marker, 1)[0]
    result = prefix + '''

def prepare(weights):
    return _expert_moe_prepare(weights, True)


def kernel_fn(data, state):
    return _expert_moe_run(data, state, True)
'''
    ast.parse(result)
    assert 'EXPERT_MOE_CAPSULE' not in result
    values = bindings(result)
    namespace = dict(values)
    node = next(n for n in ast.parse(result).body if isinstance(n, ast.FunctionDef)
                and n.name == '_expert_pair_source')
    exec(ast.get_source_segment(result, node), namespace)
    native = namespace['_expert_pair_source']()
    assert hashlib.sha256(native.encode()).hexdigest() == manifest['candidate_native_sha256']
    return result, dict(source_sha256=hashlib.sha256(result.encode()).hexdigest(),
                        bytes=len(result.encode()), native_sha256=manifest['candidate_native_sha256'],
                        paired_sha256=manifest['source_sha256'],
                        public_submission=False, target_qualified=False,
                        scope='candidate-only private packaging; local-start0/local-count32; no timing report')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))
