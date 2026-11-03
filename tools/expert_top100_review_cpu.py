"""Independent CPU/source review. Does not import the submission or use CUDA."""
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import unittest

from prod_size_compile_g import bindings
from expert_top100_pregamma import capture

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/steering/evidence/2026-11-02-top100-expert-review'
CAND = ROOT / 'tools/expert_top100_candidate.py'
BASE = ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py'
EMAX, HD = 384 * 8, 4096


def f32(x):
    try:
        return struct.unpack('<f', struct.pack('<f', x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def bf(b):
    return struct.unpack('<f', struct.pack('<I', b << 16))[0]


def rne(x):
    u = struct.unpack('<I', struct.pack('<f', f32(x)))[0]
    assert u & 0x7f800000 != 0x7f800000
    return ((u + 0x7fff + ((u >> 16) & 1)) >> 16) & 65535


class Review(unittest.TestCase):
    def test_exact_delivered_source_and_factory(self):
        a, b = BASE.read_text(encoding='utf8'), CAND.read_text(encoding='utf8')
        self.assertEqual(hashlib.sha256(a.encode()).hexdigest(), 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab')
        self.assertEqual(hashlib.sha256(b.encode()).hexdigest(), 'dff221ae6ccc2de35d7a4c123aea3e3eb1345d3685f6eb97e52cbbfef6a0c8c4')
        na, nb = bindings(a)['MOE_CUDA_SRC'], bindings(b)['MOE_CUDA_SRC']
        self.assertEqual(a.replace(na, '<NATIVE>'), b.replace(nb, '<NATIVE>'))
        ca, ra = capture(a)
        cb, rb = capture(b)
        self.assertEqual(ra, rb)
        for x, y in zip(ca, cb, strict=True):
            self.assertEqual(x['extra_cuda_cflags'], y['extra_cuda_cflags'])
            self.assertEqual(x['cpp_sources'], y['cpp_sources'])
            self.assertFalse(any('fast_math' in f or 'ftz' in f for f in y['extra_cuda_cflags']))
            if '-DOKC_WGS=1' not in y['extra_cuda_cflags']:
                self.assertEqual(x['cuda_sources'], y['cuda_sources'])
        self.assertEqual(na[na.index('#include <torch/extension.h>'):], nb[nb.index('#include <torch/extension.h>'):])

    def test_route_positions_and_nonzero_job_bases(self):
        # CPU model of lines 4162-4277; compare with independently sorted routes.
        rng = random.Random(20261102)
        for m in (32, 128, 256, 384):
            cases = [[-1] * (m * 8), [i % 8 for i in range(m * 8)],
                     [i % 32 for i in range(m * 8)],
                     [rng.randrange(-1, 40) for _ in range(m * 8)]]
            # Force skinny/fat/tail boundaries as well as unaligned job bases.
            counts = [1, 7, 8, 9, 63, 64, 65]
            boundary = [e for e, n in enumerate(counts) for _ in range(n)]
            cases.append((boundary + [-1] * (m * 8))[:m * 8])
            for ids in cases:
                expected = sorted((e, i) for i, e in enumerate(ids) if 0 <= e < 32)
                ch = ((len(ids) + 511) // 512) * 32
                wc = [[sum(ids[i] == e for i in range(w * ch, min(len(ids), (w + 1) * ch)))
                       for e in range(32)] for w in range(16)]
                counts = [sum(row[e] for row in wc) for e in range(32)]
                offsets = [sum(counts[:e]) for e in range(32)]
                pos = [-1] * len(ids)
                for w in range(16):
                    cursor = [offsets[e] + sum(wc[v][e] for v in range(w)) for e in range(32)]
                    for i in range(w * ch, min(len(ids), (w + 1) * ch)):
                        e = ids[i]
                        if 0 <= e < 32:
                            pos[i] = cursor[e]
                            cursor[e] += 1
                for q, (_, i) in enumerate(expected):
                    self.assertEqual(pos[i], q)
                jobs = []
                for fat in (True, False):
                    for e, n in enumerate(counts):
                        if n and (n > 8) == fat:
                            jobs.extend((offsets[e] + k, min(64, n - k)) for k in range(0, n, 64))
                rows = [st + t for st, n in jobs for t in range(n)]
                self.assertEqual(sorted(rows), list(range(len(expected))))
                self.assertLessEqual(len(rows), EMAX)
                self.assertTrue(all(0 < n <= 64 for _, n in jobs))

    def test_halfword_owners_alignment_and_real_regions(self):
        owners = {}
        for ub in range(16):
            for tid in range(512):
                lane, warp = tid % 32, tid // 32
                r16 = ub * 16 + warp
                for h in range(2):
                    row = 2 * (lane % 4) + h
                    for col in (r16 * 16 + lane // 4, r16 * 16 + lane // 4 + 8):
                        addr = 2 * (row * HD + col)
                        self.assertNotIn(addr, owners)
                        owners[addr] = (ub, tid)
        self.assertEqual(set(owners), set(range(0, 8 * HD * 2, 2)))
        # Two halves need not have one thread owner; both must publish before ld.cg.b32.
        self.assertNotEqual(owners[0], owners[2])
        for q in range(EMAX):
            self.assertEqual((2 * q * HD) % 4, 0)
        b4cols = [seg * 1024 + tid * 2 for seg in range(4) for tid in range(512)]
        maincols = [tid * 8 + k for tid in range(512) for k in (0, 2, 4, 6)]
        self.assertEqual(b4cols, maincols)
        self.assertTrue(all((2 * c) % 4 == 0 and c + 1 < HD for c in maincols))
        y_end = 2 * EMAX * HD
        partial_start = 4 * (EMAX - 64) * HD
        partial_end = partial_start + 4 * (32 * 8) * 256
        fix_start = 4 * EMAX * HD
        self.assertLessEqual(y_end, partial_start)
        self.assertLessEqual(partial_end, fix_start)

    def test_all_finite_raw_gamma_and_down_edges(self):
        down_bits = [0, 0x8000, 1, 0x8001, 0x7f, 0x807f, 0x80, 0x8080,
                     0x3f80, 0xbf80, 0x3f81, 0xbf81, 0x7f7f, 0xff7f]
        minimum = math.inf
        for raw in range(65536):
            if raw & 0x7f80 == 0x7f80:
                continue
            g = bf(rne(f32(1.0 + bf(raw))))
            if g:
                minimum = min(minimum, abs(g))
            for bits in down_bits:
                v = bf(bits)
                transported = bf(rne(v))
                self.assertEqual(struct.pack('<f', f32(v * g)), struct.pack('<f', f32(transported * g)))
        self.assertEqual(minimum, 2 ** -8)
        # A mixed FTZ policy would break equivalence; not evidence that current flags use it.
        self.assertNotEqual(f32(bf(1) * 1.0), 0.0)
        # Product subnormal/underflow and gamma cancellation, absent from the prior 8 gamma samples.
        self.assertEqual(bf(rne(f32(1.0 + bf(0xbf80)))), 0.0)
        self.assertEqual(f32(bf(1) * (2 ** -8)), struct.unpack('<f', struct.pack('<I', 256))[0])


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    import io
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Review))
    output = stream.getvalue()
    (OUT / 'independent-cpu.txt').write_text(output, encoding='utf8')
    print(output)
    manifest = {'evidence_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_text(encoding='utf8').encode()).hexdigest()
                          for p in (CAND, BASE, Path(__file__))},
                'tests': result.testsRun, 'success': result.wasSuccessful(), 'gpu': False,
                'actual_EMAX': EMAX, 'compact_y_end_bytes': 2 * EMAX * HD,
                'B4_partial_start_bytes': 4 * (EMAX - 64) * HD,
                'B4_live_partial_end_bytes': 4 * (EMAX - 64) * HD + 4 * 256 * 256,
                'fixup_start_bytes': 4 * EMAX * HD}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    raise SystemExit(not result.wasSuccessful())
