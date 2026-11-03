"""Finalize source-review metadata without publishing third-party code."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from intake import OUT,SCRATCH,PIN

old='636ba85648c30ae2c2bfb335c9399593a67ecc1d'
raw=(SCRATCH/'humming-version-compare.json').read_bytes();data=json.loads(raw)
metadata=dict(status=data['status'],total_commits=data['total_commits'],base=old,head=PIN,
    raw_comparison_sha256=hashlib.sha256(raw).hexdigest(),
    files=[{k:x[k] for k in ('filename','sha','status','additions','deletions','changes','raw_url') if k in x} for x in data['files']])
(OUT/'humming-version-delta.json').write_text(json.dumps(metadata,indent=2)+'\n')
reviewed=['humming/layer.py','humming/forward.py','humming/ops/gemm.py','humming/kernel/humming.py',
 'humming/schema/humming.py','humming/config/mma.py','humming/config/enum.py',
 'humming/tune/sm90.py','humming/tune/sm90_policies.py','humming/include/humming/scheduler.cuh',
 'humming/include/humming/kernel/humming_ws.cuh','humming/include/humming/datatype/dequant.cuh',
 'humming/include/humming/datatype/dequant_fused.cuh','humming/include/humming/datatype/dequant_prepare.cuh',
 'humming/include/humming/arith/mainloop_arith.cuh','humming/include/humming/arith/epilogue_arith.cuh',
 'tests/kernels/humming/test_moe.py','LICENSE']
manifest=json.loads((OUT/'source-manifest.json').read_text())
for f in manifest['files']:
    path=SCRATCH/'humming-source'/(f['path']+'.txt')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==f['sha256']
summary=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    pin=PIN,prior_pin=old,inspected_paths=reviewed,
    scratch_only_third_party_directory=str(SCRATCH),foreign_source_published=False,
    acquired_files=len(manifest['files']),source_hash_verification='PASS',
    source_acquisition_scope='All humming/ implementation and API files plus README/LICENSE/MoE test at immutable tree; focused review paths listed separately.',
    decision='NO ADEQUATE PATH FOUND',measured_savings=None)
(OUT/'decision.json').write_text(json.dumps(summary,indent=2)+'\n')
files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in OUT.iterdir() if p.is_file() and p.name!='evidence-manifest.json'}
(OUT/'evidence-manifest.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps(summary,indent=2))
