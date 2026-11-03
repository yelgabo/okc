from pathlib import Path
import json,gzip,hashlib,shutil
r=Path('/root/.cache/okc-g-short-rmw');o=Path('/mnt/c/Users/gretc/Documents/GitHub/okc-today-proj/docs/steering/evidence/2026-11-01-g-short-final-rmw-compile');o.mkdir(parents=True,exist_ok=True)
d=json.loads((r/'comparison.json').read_text())
for row in d['records']:
 assert row['resources_equal'] and row['ptx_exact_except_atomic_and_locations']
 for armname,a in zip(('control','relaxed'),row['arms']):
  a['opcode_operand_sha256']=hashlib.sha256('\n'.join(a.pop('ops')).encode()).hexdigest()
  folder=r/(row['build']+'-'+armname);dest=o/folder.name;dest.mkdir(exist_ok=True)
  for name in ('manifest.json','nvrtc.log','ptxas.log'):shutil.copyfile(folder/name,dest/name)
  for name in ('device.ptx','device.sass'):
   with gzip.GzipFile(filename=str(dest/(name+'.gz')),mode='wb',mtime=0)as f:f.write((folder/name).read_bytes())
(o/'comparison.json').write_text(json.dumps(d,indent=2)+'\n')
print('ARCHIVED',o)
