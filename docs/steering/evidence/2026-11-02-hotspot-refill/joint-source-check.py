"""Verify archived selected-code evidence without compiling or running kernels."""
import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect(root):
    path = root / 'muse_decode_layer/cand/g1b4_stage_release.py'
    text = path.read_text()
    assert sha(text.encode()) == 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
    sys.path.insert(0, str(root / 'tools'))
    from d32_b4_prefix_audit import audit
    factory = audit(text)
    helper = [r for r in factory['builds'] if '-DOKC_WS=1' in r['flags']]
    assert len(helper) == 3
    records = []
    sites = {'tiny': ('12030', '12330', '12500', '10ea0'),
             'tiny2': ('122c0', '12630', '12800', '11070'),
             'mid': ('11ae0', '11e50', '12020', '10890')}
    for name, build in zip(('tiny', 'tiny2', 'mid'), helper):
        folder = root / 'docs/steering/evidence/2026-11-02-gopt-skinny-depth'
        packed = (folder / (name + '-control.sass.gz')).read_bytes()
        asm = gzip.decompress(packed).decode()
        pp = gzip.decompress((root / 'docs/steering/evidence/2026-11-02-gopt-short-prefix' / (name + '.i.gz')).read_bytes()).decode()
        assert 'void wgs_ranges(' not in pp and 'void ws_helper(' in pp
        first, second, back, target = sites[name]
        for entry in asm.split('Function : ')[1:]:
            for address in (first, second):
                line = next(l for l in entry.splitlines() if '/*' + address + '*/' in l)
                assert 'ATOM.E.ADD.F32.FTZ.RN.STRONG.GPU' in line
            line = next(l for l in entry.splitlines() if '/*' + back + '*/' in l)
            assert 'BRA 0x' + target in line
        records.append(dict(name=name, tokens=build['tokens'], flags=build['flags'],
                            helper_sha256=build['native_sha256'], compressed_sass_sha256=sha(packed),
                            sass_sha256=sha(asm.encode()), preprocessed_sha256=sha(pp.encode()),
                            helper_fp32_atomic_offsets=[first, second], loop_backedge=[back, target]))
    folder = root / 'docs/steering/decode32/evidence/D32-ATTN-01/control-compile'
    manifest = json.loads((folder / 'manifest.json').read_text())
    data = (folder / 'device.sass').read_bytes()
    assert sha(data) == manifest['files']['device.sass']['sha256']
    insns = []
    for line in data.decode().splitlines():
        match = re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;', line)
        if match and 0x60d0 <= int(match[1], 16) <= 0x6250 and 'FMUL ' in match[2]:
            assert '.FTZ' not in match[2]
            assert match[2].split(', ')[-1].removesuffix('.reuse') == 'R24'
            insns.append(dict(offset=match[1], instruction=match[2]))
    assert len(insns) == 16
    rng = np.random.default_rng(20261102)
    bits = np.concatenate((rng.integers(0, 2**32, 1_000_000, dtype=np.uint32),
                           np.arange(65536, dtype=np.uint32) << 16,
                           np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff,
                                     0x807fffff, 0x00800000, 0x80800000], dtype=np.uint32)))
    bits = bits[(bits & 0x7f800000) != 0x7f800000]
    product = (bits.view(np.float32) * np.float32(1)).view(np.uint32)
    assert np.array_equal(bits, product)
    half_normal = np.array([0x00400000], dtype=np.uint32).view(np.float32)[0]
    subnormal_batch = np.float32(half_normal + half_normal)
    assert int(np.array([subnormal_batch]).view(np.uint32)[0]) == 0x00800000
    receipt = json.loads((root / 'muse_decode_layer/profiles/g1-stage-release.607eb52bdaf24160.json').read_text())
    rows = {r['case']['layer']: r['median_ns'] / 1000 for r in receipt['benchmarks'] if r['case']['load'] == 'decode 32'}
    c32 = 46 * rows['sliding'] + 16 * rows['full']
    return dict(evidence_base='3a961a9da35c06adb6eaf04e2531b9d323587933',
                reconciled_main='9f8fa34a991f63a26f53cccefbeade39b045f353',
                layer_sha256=sha(text.encode()), general=records,
                d32=dict(selected_instance=manifest['selected_instance'], flags=manifest['flags'],
                         cuda_sha256=manifest['source_sha256'], sass_sha256=sha(data),
                         baseline_resources=manifest['resources'], layout=manifest['layout_witness']['values'],
                         identity_rescale_sites=insns, cpu_finite_identity_cases=len(bits),
                         cpu_identity_pass=True, target_identity_status="NOT_RUN",
                         incumbent_job=receipt['job_id'], adjusted_cell_us=rows, C32_us=c32,
                         full_only_us_for_D32_1pct=c32*(1-1/1.01)/16,
                         full_only_us_for_sixload_1pct=c32*(1-1/1.01**6)/16),
                general_ftz_counterexample=dict(per_unit_input_bits='00400000',
                    two_global_FTZ_updates_from_zero='00000000', plain_register_batch_bits='00800000',
                    scope='CPU model of documented FTZ semantics, not a GPU result'),
                limitations=['Archived compiler outputs, not a new compilation',
                    'Static operation counts are not exposed time or measured gain',
                    'D32 branch occurrence and candidate resources are unmeasured',
                    'Under-295-second candidate comparison remains target-unqualified'],
                kernel_modified=False, gpu_execution=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.root)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'cpu_identity_cases': result['d32']['cpu_finite_identity_cases'],
                      'WS_builds': len(result['general']), 'D32_FMUL_sites': 16,
                      'C32_us': result['d32']['C32_us'], 'gpu_execution': False}))
