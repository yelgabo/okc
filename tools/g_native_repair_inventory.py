"""Source-only native repair ledger. Never imports/executes a submission."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

FIELDS = {'PDL', 'MOE_HAS_PDL', 'MFAB_LAYER', 'SC6_WGS', 'MOE_DEFINES',
          'MOE_TINY_DEFINES', 'MOE_TINY2_DEFINES', 'MOE_MID_DEFINES',
          'MOE_TINY_MAX_M', 'MOE_TINY2_MAX_M', 'MOE_MID_MAX_M'}


def inventory(path):
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    out = {'lf_sha256': hashlib.sha256(source.encode()).hexdigest(),
           'lf_bytes': len(source.encode()), 'base_assignments': {}, 'native_literals': []}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        name = ast.unparse(node.targets[0])
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if name in FIELDS:
            out['base_assignments'][name] = value
        if isinstance(value, str) and 'namespace okc' in value:
            out['native_literals'].append({
                'target': name, 'line': node.lineno,
                'sha256': hashlib.sha256(value.encode()).hexdigest(), 'bytes': len(value.encode()),
                'macro_definitions_in_literal': re.findall(
                    r'^#define (OKC_(?:MFIN|YBF|WGSFCI))\s+([^\n]+)', value, re.M),
                'cooperative_launch_calls': value.count('cudaLaunchCooperativeKernel('),
                'gamma_round_expressions': value.count('fork_bf16(1.f +'),
            })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    files = {'upstream': '.build/today/kel/kernels/muse_decode_layer.py',
             'incumbent': 'muse_decode_layer/cand/g1b4_stage_release.py'}
    result = {key: inventory(args.root / path) for key, path in files.items()}
    assert result['upstream']['lf_sha256'] == '82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19'
    assert result['incumbent']['lf_sha256'] == 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
    result['scope'] = ('AST literal inventory only; augmented assignments and runtime selection are '
                       'separately audited in the report. No submission execution or GPU.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
