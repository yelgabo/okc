"""Strict offline reader for this private draft; never grants dispatch/release eligibility."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import zlib
from functools import lru_cache

from expert_top100_harness_contract import positive, validate_rows

OFFICIAL = {(load, context) for load in ('decode 1','decode 16','decode 32','verify 1','verify 16','verify 32')
            for context in ('sliding','full')}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=1)
def expected_manifest():
    from dw_native_pair_harness import build
    return build()[1]


def capsule(log, manifest):
    blobs = re.findall(r'^DW_NATIVE_PAIR_REPORT ([A-Za-z0-9+/=]+)\r?$', log, re.M)
    assert len(blobs) == 1 and len(blobs[0]) < 11000, 'missing/duplicate/oversize report'
    compressed = base64.b64decode(blobs[0], validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, 1_000_000)
    assert decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data and len(raw) < 1_000_000, 'report bounds'
    def reject(value):
        raise ValueError('nonfinite JSON ' + value)
    report = json.loads(raw, parse_constant=reject)
    assert report['schema'] == 1 and report['pins'] == manifest['pins'], 'report source identity'
    assert report['official_arm'] == 'direct_weight' and report['cleanup_restored'] is True, 'official/restoration'
    assert report['math_checks'] == report['restores'] == 144 and report['fresh_checks'] == 1152, 'oracle coverage'
    assert report['transitions'] == 96, 'WGS transition coverage'
    assert positive(report['prepare_s']) < 295, 'prepare guard'
    assert len(report['owners_bytes']) == 4 and all(type(x) is int and x > 0 for x in report['owners_bytes']), 'owners'
    assert re.fullmatch('[a-f0-9]{64}', report['weight_signature']), 'weights identity'
    inventory = report['readonly_inventory']
    assert isinstance(inventory, list) and inventory, 'readonly inventory'
    for item in inventory:
        assert item['device'].startswith('cuda') and type(item['bytes']) is int and item['bytes'] > 0, 'readonly storage'
        assert item['paths'] and all(len(p) == 3 and p[0].startswith(('weights/', 'state/')) for p in item['paths']), 'readonly paths'
    result = validate_rows(report['rows'])
    assert result == report['decision'], 'printed decision mismatch'
    return report, result


def read(receipt, job, log, manifest, payload, upload, log_attestation):
    assert manifest == expected_manifest(), 'unregistered manifest'
    assert len(payload) == manifest['bytes'] and sha(payload) == manifest['source_sha256'], 'wrong source artifact'
    assert receipt['id'] == job and receipt['board'] == 'muse_decode_layer' and receipt['mode'] == 'check', 'own private job'
    assert receipt['team'] == 'yelgabo' and receipt['status'] == 'done', 'team/terminal'
    content = receipt['content_hash']
    assert re.fullmatch('[a-f0-9]{64}', content), 'server content hash'
    assert receipt['payload_ref'] == 'muse_decode_layer/' + content + '.py', 'payload ref'
    # Explicit local-upload/server association; server content_hash is NOT file SHA.
    assert upload == dict(job=job, team='yelgabo', mode='check', board='muse_decode_layer',
        content_hash=content, payload_ref=receipt['payload_ref'], source_sha256=manifest['source_sha256'], source_bytes=manifest['bytes']), 'upload association'
    owning_log = receipt['output']['log']
    assert job in Path(owning_log).name, 'owning log path'
    assert log_attestation == dict(job=job, owning_log=owning_log, sha256=sha(log.encode()),
        source_sha256=manifest['source_sha256']), 'owning log attestation'
    result = receipt['result']
    assert result['status'] == 'ok' and result['passed'] is True and result['returncode'] == 0, 'official success'
    assert result['checks']['ok'] is True and not result['checks'].get('flagged') and not result['checks'].get('rejected'), 'static checks'
    assert not result['flags'] and not result['error'], 'final error'
    records = result['records']
    tests = [r for r in records if r.get('event') == 'test']
    setup = [r for r in records if r.get('event') == 'setup']
    assert len(tests) == 13 and {r['index'] for r in tests} == set(range(13)) and all(r['status'] == 'pass' for r in tests), '13 tests'
    assert len(setup) == 1 and setup[0]['status'] == 'pass' and setup[0]['error'] is None, 'setup'
    assert positive(setup[0]['prepare_s']) < 295, 'official prepare guard'
    for event in ('benchmark', 'profile'):
        rs = [r for r in records if r.get('event') == event]
        assert len(rs) == 12 and {(r['case']['load'], r['case']['layer']) for r in rs} == OFFICIAL, '12 ' + event
        for r in rs:
            assert r['case']['weight'] == (46 if r['case']['layer'] == 'sliding' else 16), 'context weight'
            if event == 'benchmark':
                assert r['status'] == 'pass' and not r['flags'], 'benchmark failure'
                for key in ('median_ns', 'median_raw_ns', 'baseline_ns', 'baseline_raw_ns'):
                    positive(r[key])
            else:
                assert isinstance(r['report'], str) and r['report'].strip(), 'profile report'
    report, decision = capsule(log, manifest)
    return dict(job=job, source_sha256=manifest['source_sha256'], report=report, decision=decision,
        official_arm='direct_weight', dispatch_ready=False, release_eligible=False,
        scope='Held private direct-weight whole-layer instrument; CPU facades do not grant target/release qualification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('receipt','manifest','source','upload','log','log-attestation','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--job', required=True)
    a = parser.parse_args()
    load = lambda p: json.loads(p.read_text(encoding='utf8'))
    # Binary read avoids CRLF translation in the owning-log digest.
    result = read(load(a.receipt), a.job, a.log.read_bytes().decode('utf8'), load(a.manifest),
        a.source.read_bytes(), load(a.upload), load(a.log_attestation))
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
