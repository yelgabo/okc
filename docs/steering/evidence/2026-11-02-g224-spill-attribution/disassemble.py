import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
ARCHIVE = ROOT / 'docs/steering/evidence/2026-11-01-worker-g224-prefix-resource-stop'
SCRATCH = ROOT / '.build/research-g224-spill-attribution'
DISASM = Path('/root/.cache/okc-cuda130-88/cuda_nvdisasm-linux-x86_64-13.0.85-archive/bin/nvdisasm')
SCRATCH.mkdir(parents=True, exist_ok=True)
receipt = {'scope': 'Archived binaries only; disassembly/readelf; no compilation or execution', 'tools': {}, 'families': {}}
for tool in (str(DISASM), '/usr/bin/readelf'):
    receipt['tools'][tool] = {'sha256': hashlib.sha256(Path(tool).read_bytes()).hexdigest(), 'version': subprocess.run([tool, '--version'], capture_output=True, text=True, check=True).stdout}
for family in ('B4', 'MAIN'):
    manifest = json.loads((ARCHIVE / (family + '-manifest.json')).read_text())
    binary = gzip.decompress((ARCHIVE / (family + '-device.cubin.gz')).read_bytes())
    assert hashlib.sha256(binary).hexdigest() == manifest['cubin_sha256']
    target = SCRATCH / (family + '.cubin')
    target.write_bytes(binary)
    record = {'manifest': manifest, 'outputs': {}}
    for name, command in {
        'source.sass': [str(DISASM), '-g', '-gi', str(target)],
        'ptx.sass': [str(DISASM), '-gp', str(target)],
        'cfg.dot': [str(DISASM), '-bbcfg', '-poff', str(target)],
        'sections.txt': ['/usr/bin/readelf', '-SW', str(target)],
    }.items():
        result = subprocess.run(command, capture_output=True, check=True)
        raw = result.stdout
        with gzip.open(OUT / (family + '-' + name + '.gz'), 'wb') as output:
            output.write(raw)
        record['outputs'][name] = {'command': command, 'sha256': hashlib.sha256(raw).hexdigest(), 'stderr': result.stderr.decode()}
    receipt['families'][family] = record
(OUT / 'disassembly-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('Disassembled both archived cubins; four outputs per family')
