"""CPU reconstruction from archived standalone routes, NOT layer observations."""
import ast
import base64
import hashlib
import json
import re
import struct
import zlib
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[4]
raw = (root / 'muse_decode_layer/cand/g1b4_stage_release.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
source = next(ast.literal_eval(node.value) for node in ast.parse(raw.decode()).body if isinstance(node, ast.Assign) and any(getattr(target, 'id', '') == 'MOE_CUDA_SRC' for target in node.targets))
assert hashlib.sha256(source.encode()).hexdigest() == 'c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71'
costs = [int(value) for value in re.search(r'#define OKC_WGSA ([\d, ]+)', source).group(1).split(',')]
stage_path = root / 'docs/steering/evidence/2026-11-01-g-stage-probe/f147d06a64504c1e.decoded.json'
stage = json.loads(stage_path.read_text())
grid = stage['representative']['grid']
assert grid == 132


def population(counts):
    jobs = [min(64, count - offset) for count in counts if count > 8 for offset in range(0, count, 64)] + [count for count in counts if 0 < count <= 8]
    prefix = [0]
    for count in jobs:
        prefix.append(prefix[-1] + (16 * costs[(count + 7) // 8 - 1] + 160) * 32)

    def stage_at(position):
        job = max(index for index in range(len(jobs)) if prefix[index] <= position)
        cost = costs[(jobs[job] + 7) // 8 - 1]
        unit, remainder = divmod(position - prefix[job], 16 * cost + 160)
        return job * 512 + unit * 16 + (0 if remainder < 160 else min(15, (remainder - 160) // cost))

    bounds = [stage_at(prefix[-1] * cta // grid) for cta in range(grid)] + [len(jobs) * 512]
    tally = Counter()
    for lower, upper in zip(bounds, bounds[1:]):
        trailing = ((upper - 1) >> 4) << 4
        split = trailing if upper & 15 and trailing > lower else lower
        segments = []
        for start, stop in [(split, upper), (lower, split)]:
            while start < stop:
                unit, first = divmod(start, 16)
                last = min(16, first + stop - start)
                segments.append((unit, first, last))
                start += last - first
        for ordinal, (unit, first, last) in enumerate(segments):
            tally['whole' if last - first == 16 else 'split_segments'] += 1
            if last - first == 16:
                tally['whole_ntt1' if jobs[unit >> 5] <= 8 else 'whole_ntt_gt1'] += 1
                tally['whole_terminal' if ordinal == len(segments) - 1 else 'whole_nonterminal'] += 1
    return bounds, dict(tally)


receipt_path = root / 'muse_moe_w4a16/profiles/g-stage-check.f147d06a64504c1e.receipt.json'
log = json.loads(receipt_path.read_text())['result']['log_tail']
packet = next(line.partition(' ')[2] for line in log.splitlines() if line.startswith('GSP_CAPSULE '))
words = struct.unpack_from('<1900Q', zlib.decompress(base64.b64decode(packet)), 8)
bounds, tally = population(stage['representative']['endpoint0']['counts'])
for arm in range(4):
    for ordinal, cta in enumerate((0, 66, 131)):
        offset = arm * 475 + 168 + ordinal * 12
        assert tuple(words[offset:offset + 2]) == (bounds[cta], bounds[cta + 1])
rows = [dict(tokens=row[0], seed=row[1], **population(row[-1])[1]) for row in stage['report']['cases']]
for ntt in range(1, 9):
    addresses = [8 * (nt * 512 + thread) + 2 * component for nt in range(ntt) for thread in range(512) for component in range(4)]
    assert len(set(addresses)) == 2048 * ntt and max(addresses) + 2 == 4096 * ntt <= 36864
result = dict(standalone_phase_a=rows, grid=grid, representative_range_matches=12,
              scope='Six standalone inputs, NOT layer observations. Terminal = last phase-A segment per CTA.',
              compressed_mailbox_max_bytes=32768, caveat='Address/count checks only; no GPU/timing/compile/numerical validation.',
              pins={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (stage_path, receipt_path)})
(Path(__file__).parent / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
