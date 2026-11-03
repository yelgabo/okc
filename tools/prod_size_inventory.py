"""Concrete named-definition inventory; never build a mask matrix or run CUDA."""
import argparse,ast,hashlib,importlib.util,json,sys
from pathlib import Path

def load(root,name):
 p=root/'tools'/f'{name}.py';spec=importlib.util.spec_from_file_location('_size_'+name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def build(root,out):
 root=root.resolve();out.mkdir(parents=True,exist_ok=True);saved=list(sys.path)
 try:
  sys.path.insert(0,str(root/'tools'))
  gp=load(root,'dl_g_production_slice');fac=load(root,'dl_g_literal_factor');owned=load(root,'dl_owned_verify_production')
  g,gm=gp.build((root/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf8'));g,proof=fac.factor(g)
  o,om=owned.build(root,(32,))
 finally:sys.path[:]=saved
 for name,text in [('G',g),('owned32',o)]:
  (out/(name+'.py')).write_bytes(text.encode())
  for n in ast.parse(text).body:
   if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) and len(n.value.value)>1000:
    (out/(name+'-'+ast.unparse(n.targets[0]).replace("'",'')+'.cu')).write_bytes(n.value.value.encode())
 result=dict(G_bytes=len(g.encode()),owned32_bytes=len(o.encode()),G_sha=hashlib.sha256(g.encode()).hexdigest(),owned_sha=hashlib.sha256(o.encode()).hexdigest(),scope='Only source/backends, no wrapper/mask matrix, no GPU',G_manifest=gm,owned_manifest=om)
 (out/'inventory.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
 print(json.dumps({k:v for k,v in result.items() if not k.endswith('_manifest')},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();build(a.root,a.output)
