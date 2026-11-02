"""Archive fixed offline evidence and conditional score arithmetic, not timings."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from expert_top100_pregamma import ROOT


def main():
    scratch = ROOT / 'tools/expert_top100_evidence'
    out = ROOT / 'docs/steering/evidence/2026-11-02-top100-expert-prototype'
    out.mkdir(parents=True, exist_ok=True)
    audit_raw = subprocess.check_output(['git', 'show', '80eef4d:tools/top100_profile_evidence/audit.json'], cwd=ROOT)
    audit = json.loads(audit_raw)
    loads = {r['load']: r for r in audit['loads']}
    affected = ('d32', 'v16', 'v32')
    def factor(savings):
        return math.exp(sum(math.log(r['endpoint_us'] / (r['endpoint_us'] - savings.get(k, 0))) for k, r in loads.items()) / 6)
    def equal_us(target):
        lo, hi = 0., min(loads[k]['endpoint_us'] for k in affected) * .99
        for _ in range(100):
            mid = (lo + hi) / 2
            if factor({k: mid for k in affected}) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    arithmetic = dict(profile_commit='80eef4d', audit_sha256=hashlib.sha256(audit_raw).hexdigest(),
        affected=affected, unaffected=('d1', 'd16', 'v1'), loads=audit['loads'],
        one_percent_equal_affected_endpoint_us=equal_us(1.01),
        dated_cutoff_factor=audit['layer_gap_factor'], cutoff_equal_affected_endpoint_us=equal_us(audit['layer_gap_factor']),
        hypothetical_ten_percent_expert_activity=factor({k: .1 * loads[k]['expert_activity_us'] for k in affected}),
        hypothetical_ten_percent_long_full_attention_activity=factor({k: .1 * loads[k]['full_attention_activity_us'] * 16 / 62 for k in ('d16', 'd32', 'v16', 'v32')}),
        hypothetical_ten_percent_t1_expert_activity=factor({'d1': .1 * loads['d1']['expert_activity_us']}),
        warning='Conditional endpoint equations only. Activities, bytes, and compiler counts do not establish exposed savings.')
    (out / 'opportunity.json').write_text(json.dumps(arithmetic, indent=2) + '\n')
    for name in ('manifest.json', 'native.diff'):
        shutil.copyfile(scratch / 'generated' / name, out / name)
    shutil.copyfile(scratch / 'compile.json', out / 'compile.json')
    shutil.copyfile(scratch / 'generated/candidate.py', ROOT / 'tools/expert_top100_candidate.py')
    for family in ('b4', 'main'):
        for arm in ('control', 'candidate'):
            dst = out / (family + '-' + arm)
            dst.mkdir(exist_ok=True)
            for name in ('manifest.json', 'nvrtc.log', 'ptxas.log'):
                shutil.copyfile(scratch / (family + '-' + arm) / name, dst / name)
    tests = subprocess.run([sys.executable, str(ROOT / 'tools/expert_top100_test.py')], capture_output=True, text=True)
    (out / 'cpu-tests.txt').write_text(tests.stdout + tests.stderr)
    tests.check_returncode()
    hotspots = subprocess.run([sys.executable, str(ROOT / 'tools/hotspots.py'),
        str(ROOT / 'docs/steering/evidence/2026-11-02-captain-fresh-incumbent-profile/control.cc474883117947c0.json'),
        '--baseline', str(ROOT / 'muse_decode_layer/baseline_profile/profiles/baseline.7d42771c3384449d.json')], capture_output=True, text=True, encoding='utf8')
    hotspots.check_returncode()
    (out / 'hotspots.md').write_text(hotspots.stdout, encoding='utf8')
    print(json.dumps(arithmetic, indent=2))


if __name__ == '__main__':
    main()
