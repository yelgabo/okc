"""Compare compiler resources and WGMMA publication counts for this experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def inspect(folder, stem):
    log = (folder / (stem + '.log')).read_text()
    sass = (folder / (stem + '.sass')).read_text()
    entries = []
    for part in re.split(r'Function : ', sass)[1:]:
        entry = part.splitlines()[0].strip()
        if 'moe_fused' not in entry:
            continue
        waits = [len(re.findall(r'WARPGROUP\.DEPBAR\.LE gsb0, 0x' + str(n) + r'\s*;', part)) for n in (0, 1)]
        entries.append(dict(entry=entry, HGMMA=part.count('HGMMA.'), wait0=waits[0], wait1=waits[1],
                            arrive=part.count('WARPGROUP.ARRIVE'), async_fence=part.count('FENCE.VIEW.ASYNC'),
                            LDL=len(re.findall(r'\bLDL\b', part)), STL=len(re.findall(r'\bSTL\b', part))))
    resources = re.findall(r'Function properties for (\S+)\n\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', log)
    assert len(entries) == 2
    assert all(all(item[key] > 0 for key in ('HGMMA', 'wait0', 'wait1', 'arrive', 'async_fence')) for item in entries)
    return dict(entries=entries, resources=resources, sass_sha256=hashlib.sha256(sass.encode()).hexdigest())


def audit(control, candidate):
    result = {}
    for key in ('b4', 'main'):
        a, b = inspect(control, key + '-control'), inspect(candidate, key + '-candidate')
        for before, after in zip(a['entries'], b['entries']):
            assert before['entry'] == after['entry']
            for field in ('HGMMA', 'wait0', 'async_fence'):
                assert before[field] == after[field], (key, field)
            for field in ('wait1', 'arrive'):
                assert after[field] < before[field], (key, field)
            assert after['LDL'] == after['STL'] == 0
        for entry, stack, stores, loads in b['resources']:
            if 'moe_fused' in entry:
                assert (stack, stores, loads) == ('0', '0', '0')
        result[key] = dict(control=a, candidate=b)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.control, args.candidate)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: two B4 and two MAIN entries; no new spills, preserved HGMMA/drains/async-fence counts, fewer group operations')
