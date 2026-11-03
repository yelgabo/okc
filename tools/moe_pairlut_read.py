"""Validate this experiment's owning receipt with the existing strict MoE reader."""
import argparse
import hashlib
import json
from pathlib import Path

import expert_top100_moe_harness_read as reader
from moe_pairlut_harness import build
from expert_top100_moe_harness_contract import decision
from moe_pairlut_contract import pairlut_decision


def read(folder, job):
    receipt = reader.strict_json((folder / 'status-wait.raw.json').read_text())
    manifest = reader.strict_json((folder / 'paired.manifest.json').read_text())
    payload = (folder / 'paired.py').read_bytes()
    accepted = reader.strict_json((folder / 'accepted.json').read_text())
    if accepted['job_id'] != job or accepted['source_sha256'] != hashlib.sha256(payload).hexdigest():
        raise ValueError('Accepted source/job mismatch')
    if receipt.get('id') != job or receipt.get('board') != 'muse_moe_w4a16' or receipt.get('mode') != 'check':
        raise ValueError('Owning private MoE receipt mismatch')
    owning_log = receipt['output']['log']
    log_path = folder / 'profiles' / Path(owning_log).name
    log = log_path.read_bytes().decode('utf8')
    upload = dict(job=job, team='yelgabo', mode='check', board='muse_moe_w4a16',
                  content_hash=receipt['content_hash'], payload_ref=receipt['payload_ref'],
                  source_sha256=manifest['source_sha256'], source_bytes=len(payload))
    attestation = dict(job=job, owning_log=owning_log, sha256=reader.sha(log),
                       source_sha256=reader.sha(payload))
    previous = reader.build
    try:
        reader.build = lambda **kwargs: build(**kwargs)
        result = reader.read(receipt, job, log, manifest, payload, upload, attestation)
    finally:
        reader.build = previous
    if result['report']['loaded_builds']!=list(range(7)) or result['report']['table_words']!=[256,256,16640,16640]:
        raise ValueError('Observed source/lookup binding mismatch')
    verdict=pairlut_decision(result['report']['rows'],decision(result['report']['rows']))
    if result['report']['pairlut_decision']!=verdict:raise ValueError('Pair-LUT gate mismatch')
    result['pairlut_decision']=verdict
    result['target_qualified']=bool(result['target_qualified'] and verdict['passed'])
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
