"""Validate this experiment's owning receipt with the existing strict MoE reader."""
import argparse
import hashlib
import json
from pathlib import Path

import expert_top100_moe_harness_read as reader
from cuda_issue_groups_harness import build


def read(folder, job):
    receipt = reader.strict_json((folder / 'status-wait.raw.json').read_text())
    manifest = reader.strict_json((folder / 'paired.manifest.json').read_text())
    payload = (folder / 'paired.py').read_bytes()
    accepted = reader.strict_json((folder / 'accepted.json').read_text())
    if accepted['job_id'] != job or accepted['source_sha256'] != hashlib.sha256(payload).hexdigest():
        raise ValueError('Accepted source/job mismatch')
    if receipt.get('id') != job or receipt.get('board') != 'muse_moe_w4a16' or receipt.get('mode') != 'check':
        raise ValueError('Owning private MoE receipt mismatch')
    log_path = Path(receipt['output']['log'])
    log = log_path.read_bytes().decode('utf8')
    upload = dict(job=job, team='yelgabo', mode='check', board='muse_moe_w4a16',
                  content_hash=receipt['content_hash'], payload_ref=receipt['payload_ref'],
                  source_sha256=manifest['source_sha256'], source_bytes=len(payload))
    attestation = dict(job=job, owning_log=str(log_path), sha256=reader.sha(log),
                       source_sha256=reader.sha(payload))
    previous = reader.build
    treatment = {'cuda_scale_constant': 'scale', 'cuda_group_scale_fixed': 'group_scale'}[manifest['experiment']]
    try:
        reader.build = lambda **kwargs: build(treatment=treatment, **kwargs)
        result = reader.read(receipt, job, log, manifest, payload, upload, attestation)
    finally:
        reader.build = previous
    result['official_speedup'] = receipt['result']['speedup']
    result['official_speedup_ci'] = receipt['result']['speedup_ci']
    (folder / 'upload-association.json').write_text(json.dumps(upload, indent=2) + '\n')
    (folder / 'log-attestation.json').write_text(json.dumps(attestation, indent=2) + '\n')
    (folder / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    result = read(args.folder, args.job)
    print(json.dumps(dict(job=args.job, target_qualified=result['target_qualified'],
                          official_speedup=result['official_speedup'], decision=result['decision']), indent=2))
