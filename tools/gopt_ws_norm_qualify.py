"""Inspect selected helper preprocessing, compiler resources and archived controls."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

from d32_b4_prefix_audit import body
from gopt_ws_norm import ROOT, patched
from prod_size_compile_g import bindings


def qualify(folder, out):
    out.mkdir(parents=True, exist_ok=True)
    old = ROOT / 'docs/steering/evidence/2026-11-02-gopt-skinny-depth'
    helper = bindings((ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py').read_text())['_MA_ALT_SRCS']['moe_helper0']
    edited_body = body(patched(helper), '__device__ void ws_helper(')
    original_body = body(helper, '__device__ void ws_helper(')
    records = []
    for record in json.loads((folder / 'compile.json').read_text()):
        stem = record['key'] + '-' + record['arm']
        source = (folder / (stem + '.cu')).read_text()
        log = (folder / (stem + '.log')).read_text()
        sass = (folder / (stem + '.sass')).read_bytes()
        assert not re.search(r'[1-9]\d* bytes spill (stores|loads)', log)
        assert '0 bytes spill stores, 0 bytes spill loads' in log
        registers = list(map(int, re.findall(r'Used (\d+) registers', log)))
        assert registers == ([145, 145] if record['key'] == 'tiny2' else [152, 152])
        if record['arm'] == 'control':
            assert sass == gzip.decompress((old / (stem + '.sass.gz')).read_bytes())
        else:
            assert sass.count(b'FADD.FTZ') == 4
        variants = [('edited', source)]
        if record['arm'] == 'control':
            assert source.count(edited_body) == 1
            variants.append(('original', source.replace(edited_body, original_body, 1)))
        normalized = []
        for name, native in variants:
            target = stem + '-' + name
            (folder / (target + '.cu')).write_text(native)
            command = ['docker', 'run', '--rm', '--entrypoint', '', '--platform', 'linux/arm64',
                       '-v', str(folder.resolve()) + ':/w', 'nvidia/cuda:12.8.1-devel-ubuntu24.04',
                       'nvcc', *record['flags'], '-E', '/w/' + target + '.cu', '-o', '/w/' + target + '.i']
            process = subprocess.run(command, capture_output=True, text=True)
            (out / (target + '.preprocess.log')).write_text(process.stdout + process.stderr)
            process.check_returncode()
            preprocessed = (folder / (target + '.i')).read_text()
            selected_body = body(preprocessed, 'void ws_helper(')
            normalized.append(' '.join(line for line in selected_body.splitlines() if not line.lstrip().startswith('#')))
            normalized[-1] = ' '.join(normalized[-1].split())
            assert 'void wgs_ranges(' not in preprocessed
            assert ('add.rn.ftz.f32' in selected_body) == (record['arm'] == 'candidate')
            (out / (target + '.i.gz')).write_bytes(gzip.compress(preprocessed.encode(), mtime=0))
        if record['arm'] == 'control':
            assert normalized[0] == normalized[1]
        for suffix in ('.log',):
            (out / (stem + suffix)).write_bytes((folder / (stem + suffix)).read_bytes())
        for suffix in ('.sass', '.cu'):
            (out / (stem + suffix + '.gz')).write_bytes(gzip.compress((folder / (stem + suffix)).read_bytes(), mtime=0))
        record.update(registers=registers, new_spills=False, sass_sha256=hashlib.sha256(sass).hexdigest(),
                      ftz_add_sites=sass.count(b'FADD.FTZ'), fp32_atomic_sites=sass.count(b'ATOM.E.ADD.F32'),
                      control_matches_archived_ns3=record['arm'] == 'control',
                      disabled_preprocessed_body_matches_original=record['arm'] == 'control')
        records.append(record)
    (out / 'compile.json').write_text(json.dumps(records, indent=2) + '\n')
    print('PASS four selected builds; disabled helper matches original preprocessing and archived SASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    qualify(args.folder, args.out)
