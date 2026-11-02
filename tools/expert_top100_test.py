"""CPU contracts for the fixed pre-gamma transport; no CUDA/Torch import."""
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import expert_top100_pregamma as proto
from prod_size_compile_g import bindings


def f32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bf_value(bits):
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


def bf_bits(value):
    word = struct.unpack('<I', struct.pack('<f', f32(value)))[0]
    if word & 0x7f800000 == 0x7f800000:
        return (word >> 16) | (0x40 if word & 0x7fffff else 0)
    return ((word + 0x7fff + ((word >> 16) & 1)) >> 16) & 0xffff


class Contracts(unittest.TestCase):
    def test_real_factory_and_exact_source_reverse(self):
        with tempfile.TemporaryDirectory() as folder:
            m = proto.generate(Path(folder))
        self.assertEqual([r['key'] for r in m['routes']], ['tiny', 'tiny2', 'mid', 'b4', 'main', 'main'])
        self.assertEqual([r['changed'] for r in m['selection']], [False, False, False, True, True])
        self.assertLess(m['candidate_bytes'], 1000000)

    def test_all_finite_bf16_transport_and_fp32_gamma_product(self):
        # Every finite raw BF16 encoding roundtrips, including signed zero/subnormals.
        # Representative effective-gamma values cover sign, zero, subnormal and overflow.
        samples = [0, 0x8000, 1, 0x8001, 0x3f81, 0xbf81, 0x7f7f, 0xff7f]
        for bits in range(65536):
            if bits & 0x7f80 == 0x7f80:
                continue
            down = bf_value(bits)
            self.assertEqual(bf_bits(down), bits)
            transported = bf_value(bf_bits(down))
            for raw in samples:
                gamma = bf_value(bf_bits(f32(1.0 + bf_value(raw))))
                old = f32(down * gamma)
                new = f32(transported * gamma)
                self.assertEqual(struct.pack('<f', old), struct.pack('<f', new))
        # The existing post-gamma BF16 representation is NOT this treatment.
        y = f32(bf_value(0x3f81) * bf_value(0x3f81))
        self.assertNotEqual(y, bf_value(bf_bits(y)))

    def test_store_ownership_and_finalizer_address_coverage(self):
        # Actual WGS mapping: r16=ub*16+wg*4+wp; n0=r16*16+lane//4.
        # Exhaust partial N tiles and tails. Each live row/column has one writer;
        # finalizers address precisely the same compact BF16 row.
        for count in range(1, 65):
            writes = set()
            for ub in range(16):
                for tid in range(512):
                    lane, warp = tid % 32, tid // 32
                    r16 = ub * 16 + (warp // 4) * 4 + warp % 4
                    n0 = r16 * 16 + lane // 4
                    for nt in range((count + 7) // 8):
                        for h in range(2):
                            row = nt * 8 + 2 * (lane % 4) + h
                            if row < count:
                                for col in (n0, n0 + 8):
                                    addr = row * 4096 + col
                                    self.assertNotIn(addr, writes)
                                    writes.add(addr)
            self.assertEqual(writes, set(range(count * 4096)))
        for tid in range(512):
            self.assertEqual([tid * 8 + k for k in range(8)], list(range(tid * 8, (tid + 1) * 8)))
        # Compact data ends before the unchanged FP32 fixup region even at EMAX.
        self.assertLessEqual(2 * 4096 * 2048, 4 * 4096 * 2048)

    def test_protocol_math_and_negative_source_drift(self):
        old = bindings(proto.BASE.read_text())['MOE_CUDA_SRC']
        new, edits = proto.transform(old)
        for a, b, count in reversed(edits):
            new = new.replace(b, a)
        self.assertEqual(new, old)
        # The transform rejects drift in the numerical boundary, not silently patching it.
        with self.assertRaises(AssertionError):
            proto.transform(old.replace('fork_bf16(d4[h] * gsc) * w0', 'd4[h] * gsc * w0'))
        generated, _ = proto.transform(old)
        self.assertIn('__fmul_rn(v.x, g0)', generated)
        self.assertIn('static_assert(OKC_YBF == 0', generated)


if __name__ == '__main__':
    unittest.main(verbosity=2)
