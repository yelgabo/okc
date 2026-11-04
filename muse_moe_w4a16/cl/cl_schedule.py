"""Host mirror of ClSchedule (cl_schedule.h): builds the bytes the prologue kernel writes, for tests.

Policies (plan section 2.4 D2):
  "s1"   one slice per routed local expert.
  "s2"   two slices per routed local expert.
  "fill" S = floor(target / distinct) slices per routed expert (at least 1, at most 16), then the
         remaining target - S * distinct slices go one each to the experts with the most pairs
         (ties to the lower local id). target = 128 // C clusters, so 16 at C = 8 and 8 at C = 16.
An expert's 4096 intermediate columns are 16 chunks of 256; slice j of S gets 16 // S chunks, plus one
for the first 16 % S slices.
"""

import struct

MAX_SLICES = 64
MAX_EXPERTS = 32
MAX_PAIRS = MAX_EXPERTS * 8
COL_CHUNK = 256
N_CHUNKS = 16
SCHED_WORDS = 866
SCHED_BYTES = SCHED_WORDS * 4
POLICIES = {"s1": 0, "s2": 1, "fill": 2}
# 132 SMs; clusters are sized so 128 of them are used.
CLUSTER_SMS = 128

OFF = {
    "n_slices": 0, "slice_expert": 1, "slice_col0": 65, "slice_ncol": 129, "slice_tok0": 193,
    "slice_ntok": 257, "tok_list": 321, "tok_gate": 577, "expert_nslices": 833, "next_slice": 865,
}


def fill_target(C):
    return CLUSTER_SMS // C


def _slices_per_expert(cnt, policy, target):
    routed = [e for e, c in enumerate(cnt) if c > 0]
    S = [0] * len(cnt)
    if policy == "s1" or policy == "s2":
        for e in routed:
            S[e] = 1 if policy == "s1" else 2
        return S
    if policy != "fill":
        raise ValueError(policy)
    d = len(routed)
    if d == 0:
        return S
    base = min(N_CHUNKS, max(1, target // d))
    extra = target - base * d if base * d < target and base < N_CHUNKS else 0
    order = sorted(routed, key=lambda e: (-cnt[e], e))
    for r, e in enumerate(order):
        S[e] = base + (1 if r < extra else 0)
    return S


def build(topk_ids, topk_weights, first, count, C, policy, n_clusters):
    """topk_ids [T][8] int, topk_weights [T][8] float (nested lists or tensors), local range (first, count).
    Returns ClSchedule as bytes (SCHED_BYTES long)."""
    if hasattr(topk_ids, "tolist"):
        topk_ids = topk_ids.tolist()
    if hasattr(topk_weights, "tolist"):
        topk_weights = topk_weights.tolist()
    assert 0 < count <= MAX_EXPERTS
    w = [0] * SCHED_WORDS
    words_f = {}
    per_expert = [[] for _ in range(count)]
    for t, (row, wrow) in enumerate(zip(topk_ids, topk_weights)):
        for gid, g in zip(row, wrow):
            e = int(gid) - first
            if 0 <= e < count:
                per_expert[e].append((t, g))
    cnt = [len(p) for p in per_expert]
    total = sum(cnt)
    S = _slices_per_expert(cnt, policy, fill_target(C))
    n_slices = sum(S)
    w[OFF["next_slice"]] = n_clusters
    if total > MAX_PAIRS or n_slices > MAX_SLICES:
        w[OFF["n_slices"]] = -1
        return _pack(w, words_f)
    w[OFF["n_slices"]] = n_slices
    pos = 0
    i = 0
    for e in range(count):
        tok0 = pos
        for t, g in per_expert[e]:
            w[OFF["tok_list"] + pos] = t
            words_f[OFF["tok_gate"] + pos] = g
            pos += 1
        w[OFF["expert_nslices"] + e] = S[e]
        if S[e] == 0:
            continue
        q, rem = divmod(N_CHUNKS, S[e])
        for j in range(S[e]):
            w[OFF["slice_expert"] + i] = e
            w[OFF["slice_col0"] + i] = COL_CHUNK * (j * q + min(j, rem))
            w[OFF["slice_ncol"] + i] = COL_CHUNK * (q + (1 if j < rem else 0))
            w[OFF["slice_tok0"] + i] = tok0
            w[OFF["slice_ntok"] + i] = cnt[e]
            i += 1
    return _pack(w, words_f)


def _pack(w, words_f):
    out = bytearray(struct.pack("<%di" % SCHED_WORDS, *w))
    for k, g in words_f.items():
        struct.pack_into("<f", out, 4 * k, g)
    return bytes(out)


def decode(b):
    """bytes -> dict of fields, for readable diffs."""
    ints = struct.unpack("<%di" % SCHED_WORDS, b)
    floats = struct.unpack("<%df" % SCHED_WORDS, b)
    d = {}
    for name, n in (("n_slices", 1), ("slice_expert", 64), ("slice_col0", 64), ("slice_ncol", 64),
                    ("slice_tok0", 64), ("slice_ntok", 64), ("tok_list", 256), ("tok_gate", 256),
                    ("expert_nslices", 32), ("next_slice", 1)):
        src = floats if name == "tok_gate" else ints
        v = list(src[OFF[name]:OFF[name] + n])
        d[name] = v[0] if n == 1 else v
    return d
