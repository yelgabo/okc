import ast
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from attention_continuity import generate
from attention_top100_screen import LAYER, literal

OUT = Path(__file__).resolve().parent
base = (ROOT / LAYER).read_text(encoding="utf8")
candidate, original, treatment, _ = generate(base)
macros = {key: int(value) for key, value in re.findall(r"^#define (OKC_\w+) (\d+)\b", original, re.M)}
assert (macros["OKC_SEGK1"], macros["OKC_SEGK8"], macros["OKC_DYN"]) == (400, 600, 0)
for witness in (
    "const int64_t T = T0 + (SEGK > 0 ? (int64_t)(p.B - 1) * SEGK : 0);",
    "if (S.cum[mid] + mid * SEGK <= v) lo_b = mid; else hi_b = mid - 1;",
    "return min(v - lo_b * SEGK, S.cum[lo_b + 1]);",
    "cs = (int)(T * (2048 * sp + r * p.w0) / (1024LL * G));",
    "ce = (int)(T * (2048 * sp + (r ? 2048 : p.w0)) / (1024LL * G));",
):
    assert witness in original and witness in treatment, witness

def partition(lengths, query_length, weighted):
    padding = macros["OKC_SEGK1" if query_length == 1 else "OKC_SEGK8"]
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length + query_length)
    starts = [start + index * padding for index, start in enumerate(cumulative[:-1])]
    total = cumulative[-1] + (len(lengths) - 1) * padding
    def real(position):
        owner = bisect_right(starts, position) - 1
        return min(position - owner * padding, cumulative[owner + 1])
    intervals, segments = [], []
    weight = 1135 if weighted else 1024
    for index in range(264):
        sm, rank = divmod(index, 2)
        first = total * (2048 * sm + rank * weight) // (1024 * 264)
        last = total * (2048 * sm + (2048 if rank else weight)) // (1024 * 264)
        first, last = real(first), real(last)
        intervals.append((first, last))
        segments.append(sum(end > first and start < last
                            for start, end in zip(cumulative, cumulative[1:])))
    assert intervals[0][0] == 0 and intervals[-1][1] == cumulative[-1]
    assert all(first <= last for first, last in intervals)
    assert all(before[1] == after[0] for before, after in zip(intervals, intervals[1:]))
    assert sum(last - first for first, last in intervals) == cumulative[-1]
    assert sum(max(0, count - 1) for count in segments) <= len(lengths) - 1
    return dict(weight=weight, virtual_padding=padding, segments=dict(Counter(segments)),
                crossing_ctas=sum(count > 1 for count in segments),
                segment_transitions=sum(max(0, count - 1) for count in segments),
                max_segment_count=max(segments), intervals=intervals)

rng = random.Random(1102101)
samples = []
for batch in (16, 32):
    for length in (1, 8):
        vectors = [[80000] * batch, [74000 if index % 2 else 86000 for index in range(batch)]]
        vectors.extend([[rng.randrange(74000, 86001) for _ in range(batch)] for _ in range(32)])
        for weighted in ((False,) if length == 1 else (False, True)):
            rows = [partition(vector, length, weighted) for vector in vectors]
            samples.append(dict(batch=batch, query_length=length, weighted=weighted,
                cases=len(rows), crossing_ctas_min=min(row["crossing_ctas"] for row in rows),
                crossing_ctas_max=max(row["crossing_ctas"] for row in rows),
                max_segment_count=max(row["max_segment_count"] for row in rows),
                fixed_80000=rows[0]))
negative = partition([80000] * 16, 1, False)
assert negative["intervals"][0][1] != (16 * 80001 // 264)

profile_path = ROOT / "tools/top100_profile_evidence/audit.json"
profile = json.loads(profile_path.read_bytes())
loads = [row for row in profile["loads"] if row["load"] in ("d16", "d32", "v16", "v32")]
assert len(loads) == 4
def factor(saving):
    return math.exp(sum(math.log(row["endpoint_us"] /
        (row["endpoint_us"] - 16 / 62 * saving(row))) for row in loads) / 6)
def solve(target, saving):
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        if factor(lambda row: saving(row, middle)) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2
roi = []
for target in (1.01, 1.02, 1.03):
    common = solve(target, lambda row, fraction: fraction * 100) * 100
    fraction = solve(target, lambda row, fraction: fraction * row["full_attention_activity_us"])
    assert math.isclose(factor(lambda row: common), target, rel_tol=1e-12)
    roi.append(dict(layer_gain_percent=(target - 1) * 100,
        common_net_full_cell_saving_us=common,
        activity_equivalent_fraction_NOT_prediction=fraction))
result = dict(base_lf_sha256=hashlib.sha256(base.encode()).hexdigest(),
    candidate_lf_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
    candidate_lf_bytes=len(candidate.encode()),
    profile_input_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
    source_witnesses="PASS", partition_models=samples,
    padding_omission_negative="PASS: unpadded first split differs from selected padded equation",
    roi_conditional_only=roi,
    ten_percent_activity_equivalent_factor=factor(lambda row: .1 * row["full_attention_activity_us"]),
    scope="Legal deterministic synthetic lengths, not target seed reconstruction or exposed CUDA timing",
    decision="HOLD_TARGET: no demonstrated net endpoint saving budget")
OUT.joinpath("review.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
print(json.dumps({key: result[key] for key in ("candidate_lf_sha256", "candidate_lf_bytes", "roi_conditional_only", "decision")}))
