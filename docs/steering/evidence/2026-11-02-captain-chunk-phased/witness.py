import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'docs/steering/evidence/2026-11-01-worker-g2-24-route-prefix/selected.cu'
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'ed082519a4312bc1012192134b2d2865016789802b0454381984260c7aa5ca6f'
text = raw.decode()
start = text.index('__device__ void wgs_gather(')
finish = text.index('// Work left from the previous segment', start)
body = text[start:finish]
assert 'uint4 v[GB][2];' in body and 'uint4* d[GB][2];' in body
assert body.count('fence.proxy.async.global') == 1
assert body.index('fence.proxy.async.global') < body.index('CTA_SYNC();') < body.index('red.release.gpu.global.add.s32')

mapping_cases = 0
for count in range(1, 25):
    token_tiles = (count + 7) // 8
    destinations = set()
    for row in range(count):
        source_words = set()
        for thread in range(256):
            for chunk in range(2):
                column = thread + chunk * 256
                group = column & 31
                destination = ((((column >> 5) * token_tiles + (row >> 3)) * 32 + group) * 8 + (row & 7))
                assert destination not in destinations
                destinations.add(destination)
                source = (column >> 5) * 128 + 8 * ((group >> 1) & 3) + 2 * (group >> 3) + (group & 1)
                for offset in (0, 32, 64, 96):
                    assert source + offset not in source_words
                    source_words.add(source + offset)
        assert source_words == set(range(2048))
    assert len(destinations) == count * 512
    assert max(destinations) < token_tiles * 8 * 512
    mapping_cases += 1

counts = (1, 7, 8, 9, 16, 17, 23, 24)
boundaries = [0]
for count in counts:
    boundaries.append(boundaries[-1] + ((count + 7) // 8) * 8)


def event(row, chunk):
    job = next(index for index in range(len(counts)) if boundaries[index] <= row < boundaries[index + 1])
    return (row, chunk) if row - boundaries[job] < counts[job] else None


schedule_cases = 0
for first in range(0, boundaries[-1] + 1, 7):
    for length in (0, 1, 7, 8, 9, 15, 17, 33, boundaries[-1]):
        last = min(boundaries[-1], first + length)
        original = []
        for batch in range(first, last, 8):
            for row in range(batch, min(batch + 8, last)):
                for chunk in range(2):
                    actual = event(row, chunk)
                    if actual is not None:
                        original.append(actual)
        phased = []
        for chunk in range(2):
            for batch in range(first, last, 8):
                for row in range(batch, min(batch + 8, last)):
                    actual = event(row, chunk)
                    if actual is not None:
                        phased.append(actual)
        assert Counter(original) == Counter(phased)
        assert len(phased) == len(set(phased))
        release_counts = [max(0, min(last, boundaries[index + 1]) - max(first, boundaries[index])) for index in range(len(counts))]
        assert sum(release_counts) == last - first
        schedule_cases += 1

print(json.dumps(dict(scope='Source-derived mapping/schedule witness, NOT transformed-source validation, CUDA execution or emitted liveness',
    source_sha256=hashlib.sha256(raw).hexdigest(), gather_sha256=hashlib.sha256(body.encode()).hexdigest(),
    mapping_counts=mapping_cases, schedule_ranges=schedule_cases,
    current_source_dimensional_32bit_slots=dict(payload=64, pointers=32),
    proposed_source_dimensional_32bit_slots=dict(payload=32, pointers=16),
    unchanged_gather_batch=8, gpu_actions=0, performance=None), indent=2))
