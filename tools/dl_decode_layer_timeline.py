"""Decode a retained all12 timeline receipt; no GPU or network access."""
import argparse
import json
from pathlib import Path

from dl_timeline_codec import decode_report


def extract(receipt):
    if receipt.get('board') != 'muse_decode_layer':
        raise ValueError('wrong problem receipt')
    marker = 'LAYER_TIMELINE_V1 '
    lines = (receipt.get('result') or {}).get('log_tail', '').splitlines()
    blobs = [line[len(marker):] for line in lines if line.startswith(marker)]
    if len(blobs) != 1:
        raise ValueError(f'expected one retained complete report, found {len(blobs)}')
    report = decode_report(blobs[0])
    expected = {(load+layer, arm) for load in ('d1','d16','d32','v1','v16','v32')
                for layer in ('s','f') for arm in ('owned','sglang')}
    observed = [(row['case'], row['arm']) for row in report['rows']]
    if len(observed) != 24 or set(observed) != expected:
        raise ValueError('missing or duplicate shape/arm')
    report['owning_job'] = receipt['id']
    report['official_passed'] = (receipt.get('result') or {}).get('passed')
    report['routing_metadata_valid'] = all(r.get('routing_sampled_after_replay', False)
                                          for r in report['rows'])
    return report


def table(report):
    lines = [f"# Diagnostic layer timeline: {report['owning_job']}", '',
             'Cold event medians are one-input diagnostics, not official per-shape medians.',
             'CUDA activity includes device waits; union is not SM utilization.',
             'Trace coordinates:0.1us. Aggregate span/union retain original resolution.', '',
             'Recorded routing metadata valid: '+str(report['routing_metadata_valid'])+'.', '',
             '| Case | Owned cold us | SGLang cold us | Owned profile span us | SGLang profile span us | Owned uncovered us |',
             '|---|---:|---:|---:|---:|---:|']
    by = {(r['case'], r['arm']): r for r in report['rows']}
    for load in ('d1','d16','d32','v1','v16','v32'):
        for layer in ('s','f'):
            case = load+layer
            a, b = by[case,'owned'], by[case,'sglang']
            lines.append(f"|{case}|{a['cold_median_us']:.3f}|{b['cold_median_us']:.3f}|"
                         f"{a['trace']['span_us']:.3f}|{b['trace']['span_us']:.3f}|"
                         f"{a['trace']['uncovered_us']:.3f}|")
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = extract(json.loads(args.receipt.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    args.output.with_suffix('.md').write_text(table(report), encoding='utf-8')
    print(table(report))
