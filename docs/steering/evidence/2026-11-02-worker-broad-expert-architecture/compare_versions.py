"""Immutable prior/current Humming comparison; source retrieval only."""
import json
from pathlib import Path
import hashlib
from intake import get,PIN,OUT,SCRATCH

OLD='636ba85648c30ae2c2bfb335c9399593a67ecc1d'
raw,_,_=get(f'https://api.github.com/repos/vllm-project/humming/compare/{OLD}...{PIN}')
(SCRATCH/'humming-version-compare.json').write_bytes(raw)
data=json.loads(raw)
metadata=dict(status=data['status'],total_commits=data['total_commits'],base=OLD,head=PIN,
    files=[{k:x[k] for k in ('filename','sha','status','additions','deletions','changes','raw_url') if k in x} for x in data['files']])
(OUT/'humming-version-delta.json').write_text(json.dumps(metadata,indent=2)+'\n')
selected=['humming/include/humming/scheduler.cuh','humming/tune/sm90_policies.py','humming/include/humming/kernel/humming_ws.cuh','humming/forward.py','humming/schema/humming.py']
records=[]
for p in selected:
    try:
        old,_,_=get(f'https://raw.githubusercontent.com/vllm-project/humming/{OLD}/{p}')
        dest=SCRATCH/'humming-prior'/(p+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(old)
        records.append(dict(path=p,sha256=hashlib.sha256(old).hexdigest()))
    except Exception as e: records.append(dict(path=p,error=str(e)))
(OUT/'prior-source-manifest.json').write_text(json.dumps(dict(pin=OLD,files=records),indent=2)+'\n')
print(json.dumps(dict(status=data['status'],commits=data['total_commits'],files=[x['filename'] for x in data['files']],old_records=records),indent=2))
