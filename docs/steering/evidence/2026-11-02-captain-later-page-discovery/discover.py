import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
PRIOR = ROOT / 'docs/steering/evidence/2026-11-02-captain-source-refresh/retrieval.json'
QUERY = '"muse_decode_layer" NOT repo:yelgabo/okc NOT repo:lee101/openkernels NOT repo:rohanarun/Auto-research-agent'
prior = json.loads(PRIOR.read_text())
old = next(record for record in prior['records'] if record['endpoint'] == 'search/code')
known = {item['repository']['full_name'] for item in old['data']['items']}
pages = []
for page_number in (2, 3):
    command = ['gh', 'api', '-X', 'GET', 'search/code', '-f', 'q=' + QUERY,
               '-f', 'per_page=100', '-f', f'page={page_number}']
    process = subprocess.run(command, capture_output=True)
    record = {'page': page_number, 'at_utc': datetime.now(timezone.utc).isoformat(),
              'exit_code': process.returncode, 'stderr': process.stderr.decode(errors='replace'),
              'response_sha256': hashlib.sha256(process.stdout).hexdigest()}
    if process.returncode == 0:
        response = json.loads(process.stdout)
        record.update(total_count=response['total_count'],
                      incomplete_results=response.get('incomplete_results'),
                      items=[{'repository': item['repository']['full_name'],
                              'path': item['path'], 'git_blob': item['sha'],
                              'url': item['html_url']} for item in response['items']])
    pages.append(record)
all_items = [item for page in pages for item in page.get('items', [])]
repositories = sorted({item['repository'] for item in all_items})
result = {'query': QUERY, 'endpoint': 'search/code', 'pages': pages,
          'prior_page1_repositories': sorted(known),
          'later_page_repositories': repositories,
          'new_to_prior_page1': sorted(set(repositories) - known),
          'limits': ['Not a stable ranked snapshot across pages or dates.',
                     'New to one prior page is not necessarily new to all project research.',
                     'Search results are metadata, not source/performance qualification.']}
(OUT / 'discovery.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
print(json.dumps({'new_to_prior_page1': result['new_to_prior_page1'],
                  'repositories': repositories,
                  'pages': [{key: value for key, value in page.items() if key != 'items'}
                            for page in pages]}, indent=2))
