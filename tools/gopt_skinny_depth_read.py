"""Apply the existing owning-receipt checks with the skinny-depth source and gate."""
import argparse
import json
from pathlib import Path

import expert_top100_moe_harness_read as reader
from gopt_skinny_depth import build, contract_source, sha


def read(folder, job):
    receipt = reader.strict_json((folder / 'status-wait.raw.json').read_text())
    manifest = reader.strict_json((folder / 'paired.manifest.json').read_text())
    payload = (folder / 'paired.py').read_bytes()
    accepted = reader.strict_json((folder / 'accepted.json').read_text())
    if accepted['job_id'] != job or accepted['source_sha256'] != sha(payload):
        raise ValueError('Accepted source/job mismatch')
    log_path = Path(receipt['output']['log'])
    log = log_path.read_text()
    upload = dict(job=job, team='yelgabo', mode='check', board='muse_moe_w4a16',
                  content_hash=receipt['content_hash'], payload_ref=receipt['payload_ref'],
                  source_sha256=manifest['source_sha256'], source_bytes=len(payload))
    attestation = dict(job=job, owning_log=str(log_path), sha256=sha(log), source_sha256=sha(payload))
    ns = {}
    exec(contract_source(), ns)
    previous = reader.build, reader.decision
    try:
        reader.build = lambda target_proposal: build()[:2]
        reader.decision = ns['decision']
        result = reader.read(receipt, job, log, manifest, payload, upload, attestation)
    finally:
        reader.build, reader.decision = previous
    result.update(official_speedup=receipt['result']['speedup'],
                  official_speedup_ci=receipt['result']['speedup_ci'])
    for name, value in (('upload-association', upload), ('log-attestation', attestation), ('analysis', result)):
        (folder / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    (folder / 'owning.log').write_text(log)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    result = read(args.folder, args.job)
    print(json.dumps({key: result[key] for key in ('job', 'target_qualified', 'official_speedup', 'decision')}, indent=2))
