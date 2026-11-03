"""Independent stage-owning receipt reconstruction; does not import owner's analyzer."""
import argparse,ast,csv,hashlib,json,math,re
from pathlib import Path
from proj_sk_xor_release_analyze import TARGETS,LAYERS,MS,key,reconstruct
JOB='dfcfdd01d6f04525';CONTROL='9eba96894ee1402e';PIN='de1111b98072f5251ff9b2edd5d453cc22eb8fe1710a943c60922909cc796229'
def aligned(n,a):return (n+a-1)//a*a

def run(root,out):
 out.mkdir(parents=True,exist_ok=True)
 rp=root/f'muse_bf16_proj/profiles/stage-xor-release.{JOB}.receipt.json';cp=root/f'muse_bf16_proj/profiles/scored.{CONTROL}.json';sp=root/'muse_bf16_proj/experiments/sk_stage_xor_release.py'
 assert hashlib.sha256(sp.read_bytes()).hexdigest()==PIN
 r=json.loads(rp.read_text());c=json.loads(cp.read_text())
 for d,job in ((r,JOB),(c,CONTROL)):
  assert (d['id'],d['board'],d['mode'],d['status'])==(job,'muse_bf16_proj','run','done');assert d['result']['passed'] and d['result']['checks']['ok'] and not d['result']['flags']
 nr=[x for x in r['result']['records'] if x.get('event')=='benchmark'];cr=[x for x in c['result']['records'] if x.get('event')=='benchmark'];assert len(nr)==len(cr)==30
 assert math.isclose(reconstruct(nr),r['result']['speedup'],rel_tol=1e-13);assert math.isclose(reconstruct(cr),c['result']['speedup'],rel_tol=1e-13)
 new={key(x):x for x in nr};old={key(x):x for x in cr};rows=[];loads=[]
 for m in MS:
  for l in LAYERS:
   n,b=new[l,m],old[l,m];rows.append(dict(layer=l,m=m,target=(l,m) in TARGETS,new_adjusted_us=n['median_ns']/1000,C2_adjusted_us=b['median_ns']/1000,change_percent=100*(n['median_ns']/b['median_ns']-1),new_raw_us=n['median_raw_ns']/1000,C2_raw_us=b['median_raw_ns']/1000,raw_change_percent=100*(n['median_raw_ns']/b['median_raw_ns']-1),new_seed=n['case']['seed'],C2_seed=b['case']['seed']))
  loads.append(dict(m=m,new_sum5_us=sum(new[l,m]['median_ns'] for l in LAYERS)/1000,C2_sum5_us=sum(old[l,m]['median_ns'] for l in LAYERS)/1000,new_baseline_sum5_us=sum(new[l,m]['baseline_ns'] for l in LAYERS)/1000,C2_baseline_sum5_us=sum(old[l,m]['baseline_ns'] for l in LAYERS)/1000))
 log=(root/f'muse_bf16_proj/experiments/profiles/sk_stage_xor_release.{JOB}.log').read_text();resources={}
 t=ast.parse(sp.read_text(encoding='utf8'));meta=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SXS_CONFIG')
 names={('qkv_gate',8):'qkv8F',('qkv_gate',16):'qkv16F',('qkv_gate',32):'qkv32',('o',16):'o16F',('o',32):'o32F',('pooler',16):'pool16F',('pooler',32):'pool32F',('unpooler',8):'slice8F',('unpooler',16):'unp16F',('unpooler',32):'unp32F'}
 for match in re.finditer(r'^SXSRELEASE_ROUTE (\w+) (\d+) override checks=16 capacity=(\d+) arena_bytes=(\d+) resources=(\[\[.*\]\])$',log,re.M):
  layer,m,cap,arena=match.group(1),int(match[2]),int(match[3]),int(match[4]);v=ast.literal_eval(match[5]);assert len(v)==4 and all(len(x)==7 for x in v)
  cfg=meta[names[layer,m]]['original_cfg'];_,mt,tokr,nt,nwr,wsk,upw,ns,comp,hl,nacc,cl,_,fuse=cfg
  assert comp and not hl and cl==1 and not fuse
  nw=nwr*wsk;R=nwr*nt*8;units=nw*upw*nt;xby=wsk*upw*tokr*128;payload=units*768;head=aligned(units*4,16)
  assert cap<=units*512 and arena==aligned(2*cap,16)
  global_stride=aligned(xby+payload+head,1024);stage_stride=aligned(xby+payload+head+arena,1024)
  stage_dynamic=1024+ns*stage_stride+(16*ns+32)+128+16
  nesc=min(4,ns);csz=aligned((16*mt)*(R+1),4);control_zero_pmax=1024+ns*aligned(xby+payload,1024)+(16*ns+32)+4*nesc*csz+128+16+4*R
  pmax=(v[0][3]-control_zero_pmax)//8;assert pmax>=2 and pmax%2==0 and v[0][3]==control_zero_pmax+8*pmax and v[2][3]==v[0][3]
  for i,x in enumerate(v):
   assert all(type(z)is int and z>=0 for z in x) and x[1]==x[2]==0 and x[4]==x[5] and x[6]>0
   assert x[4]==32*(nw+1+(0 if i%2 else nesc))
   if i%2:assert x[3]==stage_dynamic and x[6]>=v[i-1][6]
  resources[f'{layer}:{m}']=dict(capacity=cap,arena_bytes=arena,source_NS=ns,global_SX_stride=global_stride,stage_stride=stage_stride,stage_dynamic=stage_dynamic,implied_C2_pmax=pmax,vectors=v)
 assert len(resources)==10
 done=re.search(r'^SXSRELEASE_DONE overrides=(\d+) fallbacks=(\d+) pick_keys=(\d+) checks=(\d+) captures=(\d+) fresh=(\d+)$',log,re.M);assert done and tuple(map(int,done.groups()))==(10,0,110,160,10,30)
 hybrid=[dict(b,median_ns=new[key(b)]['median_ns'] if key(b) in TARGETS else b['median_ns']) for b in cr]
 targets=[x for x in rows if x['target']];other=[x for x in rows if not x['target']]
 summary=dict(job=JOB,mode=r['mode'],source_sha256=PIN,receipt_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),control_job=CONTROL,official_score=r['result']['speedup'],reconstructed_score=reconstruct(nr),CI=r['result']['speedup_ci'],C2_official_score=c['result']['speedup'],C2_reconstructed_score=reconstruct(cr),score_ratio_to_C2=r['result']['speedup']/c['result']['speedup'],prepare_seconds=next(x['prepare_s'] for x in r['result']['records'] if x.get('event')=='setup'),tests=sum(x.get('event')=='test' and x['status']=='pass' for x in r['result']['records']),profiles=sum(x.get('event')=='profile' for x in r['result']['records']),coverage=list(map(int,done.groups())),target_adjusted_regressions=sum(x['change_percent']>0 for x in targets),target_raw_regressions=sum(x['raw_change_percent']>0 for x in targets),target_adjusted_percent_range=[min(x['change_percent'] for x in targets),max(x['change_percent'] for x in targets)],target_raw_percent_range=[min(x['raw_change_percent'] for x in targets),max(x['raw_change_percent'] for x in targets)],untargeted_faster=sum(x['change_percent']<0 for x in other),untargeted_slower=sum(x['change_percent']>0 for x in other),counterfactual_replace10_into_old20_score=reconstruct(hybrid),counterfactual_factor_to_old=reconstruct(hybrid)/c['result']['speedup'],all_stage_arenas_fit_global_SX_stride=all(v['stage_stride']==v['global_SX_stride'] for v in resources.values()),comparison_scope='Independent direct archived-C2 comparison, not paired; different seeds/weights/baselines/overhead/tuner/node may vary. Replacing ten medians is only a sensitivity model.',machine=r['result'].get('machine'),control_machine=c['result'].get('machine'))
 for name,val in (('summary',summary),('rows30',rows),('loads6',loads),('resources',resources)):(out/(name+'.json')).write_text(json.dumps(val,indent=2)+'\n')
 with (out/'rows30.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps(summary,indent=2));print('TARGETS');print('\n'.join(f"{x['layer']} {x['m']}: {x['C2_adjusted_us']:.6f}->{x['new_adjusted_us']:.6f} ({x['change_percent']:+.3f}%)" for x in targets))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.output)
