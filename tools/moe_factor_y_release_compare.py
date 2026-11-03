"""Compare one exact-file factor-y release receipt with the paired lead 7479372115c64702 per cell."""
import argparse
import json
from pathlib import Path

LEAD_US = 5698.5082429255835
LEAD_CELLS = {'decode 1': 25.667095160749792, 'decode 16': 92.39548192960114, 'decode 32': 127.21193617398296,
              'verify 1': 63.4151177930549, 'verify 16': 180.84526258384436, 'verify 32': 214.04082491104168}
SCORE_LIMIT_US = 5760.
CELL_LIMIT = .02


def compare(receipt):
    summary = receipt['summary']
    cells = {b['case']['load']: b['median_ns'] / 1000. for b in receipt['benchmarks']}
    rows = []
    for load, lead in LEAD_CELLS.items():
        actual = cells.get(load)
        change = None if actual is None else (actual - lead) / lead
        rows.append(dict(load=load, lead_us=lead, actual_us=actual, change=change,
                         within=actual is not None and change <= CELL_LIMIT))
    passed = (summary.get('passed') is True and summary.get('tests_passed') == 5 and summary.get('tests') == 5
              and len(receipt['benchmarks']) == 6 and len(receipt.get('profiles', [])) == 6)
    verdict = dict(job=receipt['job_id'], problem=receipt['problem'], passed=passed, score_us=summary.get('score_us'),
                   speedup=summary.get('speedup'), speedup_ci=summary.get('speedup_ci'), error=summary.get('error'),
                   lead_us=LEAD_US, score_within=summary.get('score_us') is not None and summary['score_us'] <= SCORE_LIMIT_US,
                   cells=rows, cells_within=all(r['within'] for r in rows))
    verdict['accepted'] = passed and verdict['score_within'] and verdict['cells_within']
    return verdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    verdict = compare(json.loads(args.receipt.read_text()))
    text = json.dumps(verdict, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text)
