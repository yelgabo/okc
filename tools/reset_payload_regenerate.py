"""Offline regeneration audit only. Never dispatches, imports CUDA payload or queues work."""
import argparse,hashlib,importlib.util,json,sys,shutil
from pathlib import Path
EXPECTED={'stage':('d32d3c8a6cccd21498250f7f27f8165b6cb8193771f78a4b66ca6453f4b5abe0',906939),'short':('8308dcf30f03ec5fec69160c6a4bb1939577b608890495a4bd018c46e926f99b',903137)}
FIXED={'stage':('aa49352105b39538b79c454f49e9f8e2014fa99b5c8205f145131169f18358e4',907822),'short':('4a1a68f5824bd51c130e03eb736a2e1fb19e011735169400b0be2c0d17448183',904020)}

def load(p,name):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def build(root,out,owners=None,owned_progress=False):
 out.mkdir(parents=True,exist_ok=True);saved=sys.path[:];sys.path.insert(0,str(root/'tools'))
 try:
  source=root/'tools'
  if owners is not None:
   source=out/'_generators';source.mkdir(exist_ok=True)
   for n in ('dl_g1_layer_gate.py','dl_g1_layer_body.py','dl_short_path_trace.py','dl_short_path_body.py'):shutil.copyfile(root/'tools'/n,source/n)
   shutil.copyfile(owners,source/'dl_g1_layer_owners.py')
  stage=load(source/'dl_g1_layer_gate.py','_reset_stage_generator')
  short=load(source/'dl_short_path_trace.py','_reset_short_generator')
  proof=stage.conformance(root/'muse_decode_layer/profiles/scored.74a29e0384324952.json')
  rows=[]
  for name in ('stage','short'):
   results=[]
   for i in range(2):
    if name=='stage':s=stage.build((root/'muse_decode_layer/cand/g1b4_control_release.py').read_text(encoding='utf8'),root,'all',proof)
    else:s=short.build((root/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf8'),root)
    raw=s.encode();pin=hashlib.sha256(raw).hexdigest();
    if owners is None:assert (pin,len(raw)) in (EXPECTED[name],FIXED[name]),(name,pin,len(raw))
    (out/f'{name}-{i}.py').write_bytes(raw);results.append(raw)
   assert results[0]==results[1];rows.append(dict(name=name,sha256=hashlib.sha256(results[0]).hexdigest(),bytes=len(results[0]),repetitions=2,identical=True))
  verify=load(root/'tools/dl_verify_pair.py','_reset_verify_generator')
  oldowner=(root/'tools/dl_g1_layer_owners.py').read_text(encoding='utf8')
  vrows=[]
  for i in range(2):
   v,vm=verify.build(root)
   oldpin=hashlib.sha256(v.encode()).hexdigest()
   if owners is not None:
    assert v.count(oldowner)==1,'four-cell shared owner source changed'
    v=v.replace(oldowner,owners.read_text(encoding='utf8'))
   if owned_progress:
    from verify_owned_progress_compose import compose
    v,progress_manifest=compose(v,root)
   raw=v.encode();(out/f'verify-{i}.py').write_bytes(raw);vrows.append(raw)
  assert vrows[0]==vrows[1]
  rows.append(dict(name='verify',sha256=hashlib.sha256(vrows[0]).hexdigest(),bytes=len(vrows[0]),repetitions=2,identical=True,old_sha256=oldpin))
  deps={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'tools'/n for n in ('dl_g1_layer_gate.py','dl_g1_layer_body.py','dl_g1_layer_owners.py','dl_g1_stage_retire.py','moe_make_sc6_compare.py','dl_make_bench_pub.py','dl_short_path_trace.py','dl_short_path_body.py','dl_timeline_codec.py')]}
  report={'scope':'Offline source-only double regeneration; payloads never imported or executed','payloads':rows,'proof':proof,'dependency_raw_sha256':deps,'owners_override_sha256':hashlib.sha256(owners.read_bytes()).hexdigest() if owners else None}
  if owned_progress:
   # Explicit opt-in preserves historical unpatched reproductions. This records
   # source composition, never runtime qualification or dispatch permission.
   report['owned_progress']=progress_manifest
   names=('dl_verify_pair.py','dl_verify_pair_body.py','dl_verify_pair_slices.py','dl_selected_pair.py','dl_selected_pair_body.py','dl_owned_slice.py','dl_owned_main_slice.py','dl_owned_attention_boundary.py','dl_g_verify_slice.py','verify_owned_progress.py','verify_owned_progress_compose.py','dl_fork_owned_cooperative.py')
   report['verify_dependency_normalized_sha256']={n:hashlib.sha256((root/'tools'/n).read_text(encoding='utf8').encode()).hexdigest() for n in names}
  (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return report
 finally:sys.path[:]=saved
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--owners',type=Path);p.add_argument('--owned-progress',action='store_true',help='Compose separately reviewed cooperative/fatal-timeout owned epoch; private check only');a=p.parse_args();build(a.root,a.output,a.owners,a.owned_progress)
