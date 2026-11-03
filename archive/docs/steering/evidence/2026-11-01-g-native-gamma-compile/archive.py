from pathlib import Path
import json,gzip,hashlib,shutil
r=Path('/root/.cache/okc-g-native-gamma');o=Path('/mnt/c/Users/gretc/Documents/GitHub/okc-today-proj/docs/steering/evidence/2026-11-01-g-native-gamma-compile');o.mkdir(parents=True,exist_ok=True)
d=json.loads((r/'analysis.json').read_text());assert d['status']=='blocked' and len(d['records'])==5
archive=[]
for row in d['records']:
 assert set(row['arms'])=={'baseline','off','on'} and all(row['off_equivalence'].values())
 for arm,a in row['arms'].items():
  folder=r/(row['build']+'-'+arm);dest=o/folder.name;dest.mkdir(exist_ok=True)
  a['normalized_sass_sha256']=hashlib.sha256(json.dumps(a.pop('sass'),sort_keys=True).encode()).hexdigest()
  for name in ('manifest.json','nvrtc.log','ptxas.log'):shutil.copyfile(folder/name,dest/name)
  for name in ('input.cu','device.ptx','device.cubin','device.sass'):
   raw=(folder/name).read_bytes()
   with gzip.GzipFile(filename=str(dest/(name+'.gz')),mode='wb',mtime=0)as f:f.write(raw)
   archive.append(dict(path=str((dest/(name+'.gz')).relative_to(o)),uncompressed_bytes=len(raw),uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
(o/'comparison.json').write_text(json.dumps(d,indent=2)+'\n')
(o/'archive.json').write_text(json.dumps(dict(builds=15,files=archive,GPU=False,compression_scope='Compiler evidence only; not entrant packaging/runtime'),indent=2)+'\n')
print('ARCHIVE',len(archive),'COMPRESSED_BYTES',sum(p.stat().st_size for p in o.rglob('*.gz')))
