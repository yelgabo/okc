import hashlib
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tracker = ROOT / 'docs/architecture-performance-tracker.md'
text = tracker.read_text()
data = json.loads((ROOT / 'docs/architecture-dashboard.json').read_text())
assert len(data['families']) == 7
assert {f['id'] for f in data['families']} == set('ABCDEFG')
header = re.search(r'Last review: (.*?)\. Evidence snapshot: `(.*?)`\.', text)
assert header, 'Tracker review header missing'
reviewed = datetime.fromisoformat(data['reviewed_at']).astimezone(ZoneInfo('America/Vancouver'))
assert header.group(1) == reviewed.strftime('%Y-%m-%d %H:%M %Z'), 'Review timestamps differ'
assert header.group(2) == data['snapshot'], 'Evidence snapshots differ'
section = text.split('## Large-token comparison ledger\n', 1)[1].split('\n## ', 1)[0]
rows = [line.strip('|').split('|') for line in section.splitlines() if line.startswith('|')][2:]
data['comparisons'] = [[cell.strip() for cell in row] for row in rows]
data['tracker_hash'] = hashlib.sha256(tracker.read_bytes()).hexdigest()[:12]
for f in data['families']:
    assert f['status'] in {'win', 'loss', 'partial', 'untested', 'inconclusive'}
    assert (ROOT / 'docs' / f['source'].split('#')[0]).is_file()
template = (ROOT / 'tools/architecture-dashboard-template.html').read_text()
output = template.replace('__DASHBOARD_DATA__', json.dumps(data).replace('<', '\\u003c'))
path = ROOT / '.lavish/architecture-performance/index.html'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(output)
print(f'Built {path}: {len(data["families"])} families, {len(rows)} comparisons, tracker {data["tracker_hash"]}')
