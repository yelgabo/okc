"""Actual generated gather contracts and CPU address/publication replay only."""
import collections
import json
from pathlib import Path
import re
import tempfile
import unittest

import g_expert_g224_chunk_phased as treatment
import g_expert_g224_prefix as parent
from test_g_expert_g224_prefix import capture, expression, source_contract

ROOT = Path(__file__).resolve().parents[1]
CONTROL = parent.transform_native(parent.literal((ROOT / parent.BASE).read_text(encoding='utf8'), 'MOE_CUDA_SRC'))
NATIVE = treatment.transform(CONTROL)


def capture_treatment(variant, modified):
    assert modified
    prefix, suffix = variant.rsplit('\nimport hashlib as _g224_chunk_hash\n', 1)
    assert parent.sha(prefix) == treatment.PARENT_VARIANT_SHA
    calls, env = capture(prefix, True)
    calls.clear()
    env['_MA_EXTS'] = {}
    env['_G224_EXTS'] = {}
    exec('\nimport hashlib as _g224_chunk_hash\n' + suffix, env)
    assert env['_G224_CUDA'] == NATIVE
    state = dict(ma_sfx=('-DOKC_PRMTSC=1',), ma_sc6=True, ma_ext={})
    for tokens in (1, 8, 16, 32, 128, 256, 384):
        env['_fork_native_moe'](tokens, state)
    return calls, env


def contract(native):
    assert source_contract(native)
    gather = treatment.body(native)
    assert gather == treatment.NEW_BODY
    assert native.replace(gather, '<GATHER>') == CONTROL.replace(treatment.OLD_BODY, '<GATHER>')
    assert gather.count('fence.proxy.async.global;') == 1
    assert gather.count('CTA_SYNC();') == 1
    tail = treatment.OLD_BODY[treatment.OLD_BODY.index('  asm volatile("fence.proxy.async.global;'):]
    assert gather.endswith(tail)
    assert '#pragma unroll 1\n  for (int chunk = 0; chunk < 2; ++chunk) {\n    int j = 0;' in gather
    assert 'uint4 v[GB];' in gather and 'uint4* d[GB];' in gather
    return True


def trace(gather, counts, q0, q1, tids=range(256)):
    job_rows, job_start, entries = [0], [], []
    for job, count in enumerate(counts):
        assert 1 <= count <= 24
        job_rows.append(job_rows[-1] + ((count + 7) // 8) * 8)
        job_start.append(len(entries))
        entries.extend((((job * 29 + row) % 384) << 3) | (row & 7) for row in range(count))
    assert 0 <= q0 <= q1 <= job_rows[-1]
    phased = 'uint4 v[GB];' in gather
    reset = gather.index('int j = 0;') > gather.index('for (int chunk = 0;') if phased else False
    assert re.search(r'for \(int qb = q0; qb < q1; qb \+= GB\)', gather)
    assert re.search(r'for \(int k = 0; k < GB; \+\+k\)', gather)
    assert re.search(r'for \(int chunk = 0; chunk < 2; \+\+chunk\)', gather)
    assert 'while (S.jrow[j + 1] <= q) ++j;' in gather
    count_match = re.search(r'const int r = (.*?), cnt = (.*?);', gather)
    row_expr, count_expr = count_match.groups()
    tile_expr, col_expr = re.search(r'const int ntt = (.*?), column = (.*?);', gather).groups()
    index_expr = re.search(r'\+ (\(+column >> 5.*?);', gather)[1]
    group_expr = re.search(r'const int group = (.*?);', gather)[1]
    input_expr = re.search(r'\+ (\(8 \* .*?);', gather)[1]
    q_expr = re.search(r'const int q = (.*?);', gather)[1]
    assert 'if (r < cnt)' in gather
    for token in ('if (q < q1)', 'S.ent[S.jS[j] + r] >> 3', '(column >> 5) * 256', '__ldg(xs32 + 32)', '__ldg(xs32 + 64)', '__ldg(xs32 + 96)'):
        assert token in gather
    def evaluate(formula, **values):
        return expression(formula.replace('S.jrow[j]', 'job_base').replace('S.jC[j]', 'job_count'), **values)
    writes = {}
    for tid in tids:
        current_job = 0
        phases = range(2) if phased else (None,)
        for phase in phases:
            if phased and reset:
                current_job = 0
            for batch in range(q0, q1, 8):
                for lane in range(8):
                    q = evaluate(q_expr, qb=batch, k=lane)
                    if q >= q1:
                        continue
                    while job_rows[current_job + 1] <= q:
                        current_job += 1
                    row = evaluate(row_expr, q=q, job_base=job_rows[current_job])
                    count = evaluate(count_expr, job_count=counts[current_job])
                    assert row >= 0
                    if row >= count:
                        continue
                    for chunk in (phase,) if phased else range(2):
                        column = evaluate(col_expr, tid=tid, chunk=chunk, NT=256)
                        tiles = evaluate(tile_expr, cnt=count)
                        offset = evaluate(index_expr, column=column, ntt=tiles, r=row)
                        group = evaluate(group_expr, column=column)
                        word_base = evaluate(input_expr, group=group)
                        columns = tuple((column >> 5) * 256 + 2 * (word_base + component * 32) + half for component in range(4) for half in range(2))
                        token = entries[job_start[current_job] + row] >> 3
                        destination = job_rows[current_job] * 512 + offset
                        assert destination not in writes
                        writes[destination] = (current_job, row, token, columns)
    releases = {}
    cursor, job = q0, 0
    while cursor < q1:
        while job_rows[job + 1] <= cursor:
            job += 1
        end = min(q1, job_rows[job + 1])
        releases[job] = end - cursor
        cursor = end
    return writes, releases, job_rows


class ChunkPhasedContracts(unittest.TestCase):
    def test_exact_parent_reverse_real_entry_and_fixed_flags(self):
        self.assertTrue(contract(NATIVE))
        self.assertEqual(NATIVE.replace(treatment.NEW_BODY, treatment.OLD_BODY), CONTROL)
        with tempfile.TemporaryDirectory() as directory:
            variant, native, manifest = treatment.generate(ROOT, Path(directory))
            self.assertEqual(native, NATIVE)
            control_variant = (Path(directory) / 'control/variant.py').read_text()
            self.assertTrue(variant.startswith(control_variant))
            old_calls, _ = capture(control_variant, True)
            new_calls, _ = capture_treatment(variant, True)
            self.assertEqual(len(old_calls), len(new_calls))
            selected = 0
            for old, new in zip(old_calls, new_calls):
                self.assertEqual(old['extra_cuda_cflags'], new['extra_cuda_cflags'])
                self.assertEqual(old['cpp_sources'], new['cpp_sources'])
                if '-DOKC_WGS=1' in new['extra_cuda_cflags']:
                    selected += 1
                    self.assertEqual(old['cuda_sources'], CONTROL)
                    self.assertEqual(new['cuda_sources'], NATIVE)
                else:
                    self.assertEqual(old, new)
            self.assertEqual(selected, 2)
            self.assertTrue(manifest['exact_reverse'])

    def test_all_counts_all_compute_threads_full_row_bijection(self):
        for count in range(1, 25):
            end = ((count + 7) // 8) * 8
            changed, releases, _ = trace(treatment.NEW_BODY, [count], 0, end)
            control, old_releases, _ = trace(treatment.OLD_BODY, [count], 0, end)
            self.assertEqual(changed, control)
            self.assertEqual(releases, old_releases)
            self.assertEqual(releases, {0: end})
            rows = collections.defaultdict(list)
            for job, row, token, columns in changed.values():
                rows[job, row].extend(columns)
                self.assertEqual(token, row)
            self.assertEqual(set(rows), {(0, row) for row in range(count)})
            for columns in rows.values():
                self.assertEqual(len(columns), 4096)
                self.assertEqual(set(columns), set(range(4096)))
            self.assertEqual(len(changed), count * 512)

    def test_partial_batches_inside_job_padding_multiple_jobs_and_empty(self):
        counts = [1, 8, 9, 16, 17, 24, 3]
        total = sum(((count + 7) // 8) * 8 for count in counts)
        for begin, end in ((0, 0), (0, 1), (1, 7), (3, 13), (7, 9), (9, 22), (20, 29), (total - 7, total), (total, total), (0, total)):
            changed, releases, rows = trace(treatment.NEW_BODY, counts, begin, end)
            control, old_releases, _ = trace(treatment.OLD_BODY, counts, begin, end)
            self.assertEqual(changed, control)
            self.assertEqual(releases, old_releases)
            self.assertEqual(sum(releases.values()), end - begin)
            expected_rows = sum(1 for job, count in enumerate(counts) for row in range(count) if begin <= rows[job] + row < end)
            self.assertEqual(len(changed), expected_rows * 512)
        self.assertEqual(trace(treatment.NEW_BODY, [], 0, 0)[0:2], ({}, {}))

    def test_lifetime_publication_and_j_reset_negatives(self):
        for old, new in (('uint4 v[GB];', 'uint4 v[GB][2];'), ('constexpr int GB = OKC_GATHB;', 'constexpr int GB = 4;'),
                         ('column = tid + chunk * NT', 'column = tid'), ('if (r < cnt)', 'if (r <= cnt)'),
                         ('#pragma unroll 1', '#pragma unroll'), ('"r"(end - q)', '"r"(2 * (end - q))'),
                         ('  CTA_SYNC();', ''), ('        if (d[k]) *d[k] = v[k];', '        if (d[k]) *d[k] = v[k];\n  CTA_SYNC();')):
            with self.assertRaises(AssertionError, msg=old):
                contract(NATIVE.replace(old, new, 1))
        stale = treatment.NEW_BODY.replace('    int j = 0;\n', '', 1).replace('#pragma unroll 1\n', '  int j = 0;\n#pragma unroll 1\n', 1)
        with self.assertRaises(AssertionError):
            trace(stale, [3, 9, 24], 0, 48, tids=(0,))
        with self.assertRaises(AssertionError):
            contract(NATIVE.replace(treatment.NEW_BODY, stale, 1))


if __name__ == '__main__':
    unittest.main()
