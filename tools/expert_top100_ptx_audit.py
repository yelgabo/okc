"""Inspect existing pinned secondary PTX; never compile or execute CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def inspect(folder):
    records = []
    for family in ('b4', 'main'):
        manifest = json.loads((ROOT / 'docs/steering/evidence/2026-11-02-top100-expert-prototype' /
                               (family + '-candidate') / 'manifest.json').read_text())
        assert manifest['source_sha256'] == '3cc12ae09f1954013f5cc9be858120d29b12bf14fff267391b94c5000d2256ea'
        ptx = (folder / (family + '-candidate') / 'device.ptx').read_bytes()
        assert hashlib.sha256(ptx).hexdigest() == manifest['ptx_sha256']
        entries = {}
        entry, loc = None, ''
        for number, line in enumerate(ptx.decode().splitlines(), 1):
            match = re.search(r'\.(?:visible|weak) \.entry (\S+)\(', line)
            if match:
                entry = match[1]
                entries[entry] = dict(products=[], stores=[], product_other=[])
            if line.lstrip().startswith('.loc'):
                loc = line
            if entry is None:
                continue
            # Source lines are pinned by the compile manifest, including its adapter.
            if re.search(r'\.loc\s+1 2027 ', loc):
                if re.search(r'\bmul\.rn\.f32\b', line):
                    entries[entry]['products'].append(dict(ptx_line=number, instruction=line.strip()))
                elif re.search(r'\b(?:mul|fma)\.', line):
                    entries[entry]['product_other'].append(dict(ptx_line=number, instruction=line.strip()))
            if re.search(r'inlined_at 1 199[34] ', loc) and 'st.global' in line:
                assert '.b16 ' in line, line
                entries[entry]['stores'].append(dict(ptx_line=number, instruction=line.strip()))
        assert set(entries) == set(manifest['lowered'])
        for evidence in entries.values():
            assert evidence['products'] and evidence['stores'] and not evidence['product_other']
        records.append(dict(family=family, ptx_sha256=manifest['ptx_sha256'],
                            native_sha256=manifest['source_sha256'],
                            compiled_prefix_sha256=manifest['compiled_device_prefix_sha256'],
                            entries={name: dict(product_count=len(e['products']), store_count=len(e['stores']),
                                product_examples=e['products'][:2], store_examples=e['stores'][:2],
                                product_other=e['product_other']) for name, e in entries.items()}))
    return dict(decision='PINNED_SECONDARY_PTX_BOUNDARIES_PASS', records=records,
                scope='NVRTC12.8 PTX only; not target NVCC, SASS equivalence, runtime numerics or timing',
                source_lines=dict(reconstructed_product=2027, compact_store_calls=[1993, 1994]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.artifacts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(result['decision'])
