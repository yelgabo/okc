import base64
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SCRATCH = ROOT / '.build/captain-nvfp4-engine-intake'
REPO = 'axolotl-ai-cloud/axolotl'
PIN = '58deaee13ce0625e0b431ece379b91c7715b8aae'
PREFIX = 'src/axolotl/integrations/kernels/libs/'
PATHS = [
    'docs/nvfp4_lora.qmd',
    PREFIX + 'sonicmoe/nvfp4.py',
    PREFIX + 'sonicmoe/nvfp4_lora.py',
    PREFIX + 'sonicmoe/nvfp4_quant.py',
    PREFIX + 'sonicmoe/triton_nvfp4.py',
    PREFIX + 'sonicmoe/fp4_cute.py',
    PREFIX + 'scattermoe_lora/marlin_w4a16/backend.py',
    PREFIX + 'scattermoe_lora/marlin_w4a16/prep.py',
    'src/axolotl/integrations/kernels/adapters/nvfp4_moe.py',
]

def github(endpoint):
    process = subprocess.run(['gh', 'api', endpoint], capture_output=True)
    if process.returncode:
        return None, process.stderr.decode('utf-8', errors='replace')
    return json.loads(process.stdout), None

SCRATCH.mkdir(parents=True, exist_ok=True)
records = []
for source_path in PATHS:
    endpoint = f'repos/{REPO}/contents/{source_path}?ref={PIN}'
    response, error = github(endpoint)
    if error:
        records.append({'path': source_path, 'endpoint': endpoint, 'error': error})
        continue
    content = base64.b64decode(response['content'])
    blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
    if blob != response['sha']:
        raise ValueError(f'Git blob mismatch: {source_path}')
    destination = SCRATCH / source_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    records.append({'path': source_path, 'endpoint': endpoint, 'bytes': len(content),
                    'git_blob': blob, 'sha256': hashlib.sha256(content).hexdigest()})
result = {'retrieved_utc': datetime.now(timezone.utc).isoformat(),
          'repository': REPO, 'commit': PIN, 'files': records,
          'limitations': ['Foreign source downloaded only, never imported or executed.',
                         'Node-runtime gh authentication failed; shell-runtime gh used.',
                         'Selected file intake, not a complete repository audit.']}
(OUT / 'intake.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
