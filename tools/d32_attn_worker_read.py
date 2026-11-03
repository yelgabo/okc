"""Validate the private attention pair report against its registered contract."""
import argparse
import base64
import json
import zlib
from pathlib import Path

from d32_attn_worker_contract import decision
from dl_compare_receipts import validate

PREFIX = 'D32_ATTENTION_PAIR_REPORT '
CANDIDATE = '5a580c7a0fad1246503ebc66cfbeca698c2b011638070f340480e4560a18d3b5'
CONTROL = 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
CORE = '302986ababc1d75286258715133dcd9a98802bd05dd71e5b251df4e041deddbd'


def report(log):
    capsules = [line[len(PREFIX):] for line in log.splitlines() if line.startswith(PREFIX)]
    if len(capsules) != 1:
        raise ValueError('expected exactly one matched report')
    data = json.loads(zlib.decompress(base64.b64decode(capsules[0], validate=True)))
    if (data.get('pins', {}).get('candidate') != CANDIDATE or data['pins'].get('control') != CONTROL
            or data['pins'].get('harness_core_sha256') != CORE):
        raise ValueError('candidate, control, or harness identity mismatch')
    if (data.get('official_arm') != 'attention_treatment' or data.get('cleanup_restored') is not True
            or (data.get('math_checks'), data.get('fresh_checks'), data.get('restores'),
                data.get('transitions')) != (144, 1152, 144, 96)):
        raise ValueError('official arm, cleanup, or validation coverage mismatch')
    for row in data['rows']:
        target = row['tokens'] == 32 and row['context'] == 'full'
        if row.get('selected') != [[False], [False], [target], [target]]:
            raise ValueError('actual attention selection mismatch')
    outcome = decision(data['rows'])
    if data.get('decision') != outcome:
        raise ValueError('matched decision does not recompute')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    verified = validate(receipt, args.job, 'check')
    paired = report(receipt['result']['log_tail'])
    print(json.dumps({'job': args.job, 'official_score': verified['score'],
                      'paired_decision': paired['decision'], 'rows': paired['rows'],
                      'prepare_s': paired.get('prepare_s'),
                      'qualification': 'MATCHED_PRIVATE_SCREEN_NOT_PUBLIC_PROMOTION'}, indent=2))


if __name__ == '__main__':
    main()
