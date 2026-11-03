import ctypes
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from prod_size_compile_g import bindings

raw = (ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
source = bindings(raw.decode())['MOE_CUDA_SRC']
assert hashlib.sha256(source.encode()).hexdigest() == 'c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71'
checks = {
    'three_weight_slots': 'constexpr int WGS_NS = 3;',
    'sc6_slot_stride': 'UNIT_BYTES = UNIT_CODES + UNIT_SC;',
    'per_thread_empty_count': 'k < BX_F ? NT : k < BX_E ? 1 : NT',
    'per_stage_activation_retire': '// All reads of this activation slot are complete before the ring can reuse it.',
    'proxy_shared_fence': 'fence.proxy.async.shared::cta;',
    'proxy_global_fence': 'fence.proxy.async.global;',
    'deferred_flush_group_four': '#define OKC_WGSFCI 4',
    'two_k16_steps_per_group': '#define OKC_WGSGS 2',
    'weight_producer_early_lane_exit': 'if ((threadIdx.x & 31) != 0) return;',
    'selected_mfin_disabled': '#if OKC_WGS\n#undef OKC_MFIN\n#define OKC_MFIN 0',
}
for name, literal in checks.items():
    assert literal in source, name

class SmemMirror(ctypes.Structure):
    _fields_ = [
        ('lut', ctypes.c_uint32 * 256), ('red', ctypes.c_float * 4),
        ('bars', ctypes.c_uint64 * 24), ('wglast', ctypes.c_int * 4),
        ('wcp', ctypes.c_int * 89), ('wcpB', ctypes.c_int * 89), ('jrow', ctypes.c_int * 89),
        ('wtot', ctypes.c_uint32), ('wtotB', ctypes.c_uint32), ('wgr', ctypes.c_int * 4),
        ('seq', ctypes.c_int), ('fseq', ctypes.c_int), ('preissued', ctypes.c_int), ('preissuedB', ctypes.c_int),
        ('fcnt', ctypes.c_int * 32), ('xbuf_ptr', ctypes.c_void_p),
        ('cnt', ctypes.c_int * 64), ('off', ctypes.c_int * 64), ('wc', ctypes.c_int * 512),
        ('jE', ctypes.c_int * 88), ('jS', ctypes.c_int * 88), ('jC', ctypes.c_int * 88), ('forder', ctypes.c_int * 88),
        ('njF', ctypes.c_int), ('njS', ctypes.c_int), ('unit', ctypes.c_int), ('gen', ctypes.c_int),
        ('ssp', ctypes.c_void_p), ('uslot', ctypes.c_int * 8), ('dync', ctypes.c_int * 88),
        ('unit_next', ctypes.c_int), ('unit_next2', ctypes.c_int),
        ('eid', ctypes.c_byte * 3072), ('ent', ctypes.c_short * 3072), ('pos', ctypes.c_short * 3072),
    ]

base = (ctypes.sizeof(SmemMirror) + 127) // 128 * 128
assert base == 22272
weights = [[base + slot * 36864, base + (slot + 1) * 36864] for slot in range(3)]
activations = [[base + 3 * 36864 + slot * 32768, base + 3 * 36864 + (slot + 1) * 32768] for slot in range(3)]
assert activations[-1][1] == 231168

def interleavings(left, right):
    if not left:
        yield right
    elif not right:
        yield left
    else:
        for suffix in interleavings(left[1:], right):
            yield left[:1] + suffix
        for suffix in interleavings(left, right[1:]):
            yield right[:1] + suffix

traces = 0
for last_sequence in range(48):
    slot = last_sequence % 3
    assert (last_sequence + 3) % 3 == slot
    for ntt in range(1, 9):
        for ordering in interleavings(['helper_read', 'helper_epi', 'helper_release'], ['next_ci0', 'next_ci1', 'next_ci2', 'next_ci3', 'next_ci4']):
            released = False
            for event in ordering:
                if event == 'helper_release':
                    released = True
                if not released:
                    assert (last_sequence + 1) % 3 != slot
                    assert (last_sequence + 2) % 3 != slot
            assert released
            traces += 1
        addresses = [16 * thread + 4 * component for thread in range(512) for component in range(4)]
        assert len(set(addresses)) == 2048 and max(addresses) + 4 == 8192

result = dict(
    source_sha256=hashlib.sha256(raw).hexdigest(), cuda_sha256=hashlib.sha256(source.encode()).hexdigest(),
    checks=checks, smem_mirror_size=ctypes.sizeof(SmemMirror), ring_base=base,
    weight_slots=weights, activation_slots=activations,
    mailbox_by_slot=[[start, start + 8192] for start, end in weights],
    spare_completion_mbar_offset=SmemMirror.bars.offset + 12 * 8,
    reserved_sequence_scenarios=48 * 8, abstract_interleavings=traces,
    unsafe_trace=['all 512 original BW_E arrivals', 'producer issues sequence q+3 to same slot', 'compute stores mailbox into same slot'],
    streaming_invariant='NTT-1 fragments must be consumed before final fragment can replace them; only final fragment can survive compute entering next unit.',
    overlap_deadline='For unchanged pending doneA release, next segment first-stage ci4 wgs_flush must acquire helper completion before original release.',
    disclaimer='CPU address/event consistency only. Mirror is not NVCC ABI measurement; event enumeration assumes required barriers/fences. Not a PTX memory-model proof, compiler/resource result, GPU correctness or timing evidence.')
(Path(__file__).parent / 'source-excerpts.txt').write_text('\n\n'.join(
    source[source.index(start):source.index(end, source.index(start))]
    for start, end in [
        ('constexpr int WGS_NS', 'constexpr int WGS_CA'),
        ('__device__ void wgs_wproducer', '// phase A\'s x gather'),
        ('struct WgsPend', '// cost-weighted stage ranges'),
    ]))
print(json.dumps(result, indent=2))
