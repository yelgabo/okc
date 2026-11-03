import argparse
import json
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--skill-root', type=Path, default=Path.home() / '.codex/skills/archify')
parser.add_argument('--review', default='review-neutral')
args = parser.parse_args()
folder = Path(__file__).resolve().parent
repo = folder.parents[2]
bundle = repo / '.build' / ('archify-neutral-' + folder.name)
if bundle.exists():
    shutil.rmtree(bundle)
version = json.loads((args.skill_root / 'package.json').read_text())['version']
if version != '3.2.0':
    raise RuntimeError(f'Expected Archify 3.2.0, found {version}')
shutil.copytree(args.skill_root, bundle, ignore=shutil.ignore_patterns('.git', 'node_modules', '__pycache__'))
template = bundle / 'assets/template.html'
css = (folder / 'neutral.css').read_text()
template.write_text(template.read_text().replace('</head>', f'<style id="okc-neutral">\n{css}\n</style>\n</head>'))
for candidate in sorted(folder.glob('*.candidate.json')):
    name = candidate.name.removesuffix('.candidate.json')
    candidate = folder / f'{name}.candidate.json'
    data = json.loads(candidate.read_text())
    output = repo / data['meta']['output']
    subprocess.run(['node', str(bundle / 'bin/archify.mjs'), 'finalize', 'architecture', str(candidate), str(output), '--repo-root', str(repo), '--quality', 'showcase', '--out-dir', str(folder / f'{name}-{args.review}'), '--json'], check=True, cwd=repo)
