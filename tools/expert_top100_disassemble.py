"""Archive actual compiled instruction census; counts are not performance."""
import collections
import hashlib
import json
import subprocess

from expert_top100_pregamma import ROOT
from prod_size_compile_g import sections


def main():
    result = {}
    for family in ('b4', 'main'):
        result[family] = {}
        for arm in ('control', 'candidate'):
            folder = ROOT / 'tools/expert_top100_evidence' / (family + '-' + arm)
            disasm = subprocess.check_output(['/root/.cache/okc-cuda128/cuda_nvdisasm-linux-x86_64-12.8.90-archive/bin/nvdisasm', str(folder / 'device.cubin')])
            path = folder / 'device.sass'
            path.write_bytes(disasm)
            functions = sections(path)
            census = {}
            for name, ops in functions.items():
                if 'moe_fused' not in name:
                    continue
                counts = collections.Counter(op.split()[1] if op.startswith('@') else op.split()[0] for op in ops)
                waits = collections.Counter(op for op in ops if 'WARPGROUP.DEPBAR' in op)
                census[name] = dict(instructions=len(ops), wait_instructions=dict(waits), selected_counts={k: v for k, v in counts.items() if any(t in k for t in ('HGMMA', 'WARPGROUP', 'MEMBAR', 'BAR', 'LDL', 'STL'))})
            result[family][arm] = dict(sass_sha256=hashlib.sha256(disasm).hexdigest(), entries=census)
        for name, old in result[family]['control']['entries'].items():
            new = result[family]['candidate']['entries'][name]
            assert old['wait_instructions'] == new['wait_instructions']
            for key, count in old['selected_counts'].items():
                if not any(t in key for t in ('LDL', 'STL')):
                    assert new['selected_counts'].get(key) == count, (family, name, key)
    out = ROOT / 'docs/steering/evidence/2026-11-02-top100-expert-prototype/sass-census.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
