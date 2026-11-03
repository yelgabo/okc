"""Terminal owning-receipt analysis of single-arm public SX result; no paired inference."""
import argparse,ast,csv,hashlib,json,math,re
from pathlib import Path
JOB='2b537518091248f1';CONTROL='9eba96894ee1402e';SOURCE='685ffab5d1614343d3c46d505967ae6817e3784736297018c78984354333f765'
TARGETS={('qkv_gate',8),('qkv_gate',16),('qkv_gate',32),('o',16),('o',32),('pooler',16),('pooler',32),('unpooler',8),('unpooler',16),('unpooler',32)}
LAYERS=('qkv_gate','o','pooler','unpooler','router');MS=(1,8,16,32,128,256)
def key(r):return r['case']['layer'],r['case']['m']
def reconstruct(rows,field='median_ns'):
 by={key(r):r for r in rows};assert all(r['case']['weight']==62 and math.isfinite(r[field]) and r[field]>0 and math.isfinite(r['baseline_ns']) and r['baseline_ns']>0 for r in rows);assert set(by)=={(l,m) for l in LAYERS for m in MS}
 return math.prod(sum(by[l,m]['baseline_ns'] for l in LAYERS)/sum(by[l,m][field] for l in LAYERS) for m in MS)**(1/6)
def analyze(root,out):
 out.mkdir(parents=True,exist_ok=True)
 rp=root/f'muse_bf16_proj/profiles/sk-xor-release.{JOB}.receipt.json';cp=root/f'muse_bf16_proj/profiles/scored.{CONTROL}.json'
 r=json.loads(rp.read_text());c=json.loads(cp.read_text());assert (r['id'],r['board'],r['mode'])==(JOB,'muse_bf16_proj','run');assert r['result']['passed'] and r['result']['checks']['ok'] and not r['result']['flags']
 source=root/'muse_bf16_proj/experiments/sk_sparse_xor_release.py';assert hashlib.sha256(source.read_bytes()).hexdigest()==SOURCE
 assert (c['id'],c['board'],c['mode'])==(CONTROL,'muse_bf16_proj','run')
 result=r['result'];old=c['result'];rs=[b for b in result['records'] if b.get('event')=='benchmark'];cs=[b for b in old['records'] if b.get('event')=='benchmark'];assert len(rs)==len(cs)==30
 setup=next(b for b in result['records'] if b.get('event')=='setup');tests=[b for b in result['records'] if b.get('event')=='test'];assert len(tests)==9 and all(t['status']=='pass' for t in tests)
 assert math.isclose(reconstruct(rs),result['speedup'],rel_tol=1e-13);assert math.isclose(reconstruct(cs),old['speedup'],rel_tol=1e-13)
 by={key(b):b for b in rs};cb={key(b):b for b in cs};pnew={key(b):b['report'] for b in result['records'] if b.get('event')=='profile'};pold={key(b):b['report'] for b in old['records'] if b.get('event')=='profile'};assert len(pnew)==30
 rows=[]
 for m in MS:
  for l in LAYERS:
   k=l,m;n,b=by[k],cb[k];nu=n['median_ns']/1000;bu=b['median_ns']/1000
   rows.append(dict(layer=l,m=m,target=k in TARGETS,new_seed=n['case']['seed'],old_seed=b['case']['seed'],new_adjusted_us=nu,old_adjusted_us=bu,delta_us=nu-bu,latency_change_percent=100*(nu/bu-1),new_raw_us=n['median_raw_ns']/1000,old_raw_us=b['median_raw_ns']/1000,new_offset_us=(n['median_raw_ns']-n['median_ns'])/1000,old_offset_us=(b['median_raw_ns']-b['median_ns'])/1000,new_baseline_us=n['baseline_ns']/1000,old_baseline_us=b['baseline_ns']/1000,new_profile=pnew[k],old_profile=pold[k]))
 log=(root/f'muse_bf16_proj/experiments/profiles/sk_sparse_xor_release.{JOB}.log').read_text()
 resource={}
 for m in re.finditer(r'SXRELEASE_ROUTE\s+(\w+)\s+(\d+)\s+override\s+checks=(\d+)\s+resources=(\[\[.*?\]\])',log,re.S):
  vectors=ast.literal_eval(m[4]);assert len(vectors)==4 and all(len(v)==7 and v[1]==0 for v in vectors)
  resource[m[1]+':'+m[2]]={'checks':int(m[3]),'order':['control','candidate','control_nf','candidate_nf'],'fields':['registers','local_bytes','static_shared_bytes','dynamic_shared_bytes','threads','max_threads','resident_blocks_per_SM_admission'],'vectors':vectors}
 assert len(resource)==10
 coverage=re.search(r'SXRELEASE_DONE overrides=(\d+) fallbacks=(\d+) pick_keys=(\d+) checks=(\d+) captures=(\d+) fresh=(\d+)',log);assert coverage and tuple(map(int,coverage.groups()))==(10,0,110,160,10,30)
 loads=[];hybrid=[]
 for k,b in cb.items():
  v=dict(b);v['median_ns']=by[k]['median_ns'] if k in TARGETS else b['median_ns'];hybrid.append(v)
 for m in MS:
  nsum=sum(by[l,m]['median_ns'] for l in LAYERS)/1000;bsum=sum(cb[l,m]['median_ns'] for l in LAYERS)/1000;hsum=sum((by if (l,m) in TARGETS else cb)[l,m]['median_ns'] for l in LAYERS)/1000
  loads.append({'m':m,'candidate_sum_us':nsum,'archived_C2_sum_us':bsum,'candidate_baseline_sum_us':sum(by[l,m]['baseline_ns'] for l in LAYERS)/1000,'archived_baseline_sum_us':sum(cb[l,m]['baseline_ns'] for l in LAYERS)/1000,'candidate_official_load_ratio':sum(by[l,m]['baseline_ns'] for l in LAYERS)/(nsum*1000),'archived_official_load_ratio':sum(cb[l,m]['baseline_ns'] for l in LAYERS)/(bsum*1000),'counterfactual_replace10_sum_us':hsum,'counterfactual_factor_vs_archived_load':bsum/hsum})
 target=[r for r in rows if r['target']];other=[r for r in rows if not r['target']]
 summary={'job':JOB,'board':r['board'],'mode':r['mode'],'source_sha256':SOURCE,'source_bytes':source.stat().st_size,'receipt_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'control_job':CONTROL,'control_receipt_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'official_score':result['speedup'],'official_ci':result['speedup_ci'],'reconstructed_score':reconstruct(rs),'archived_C2_official_score':old['speedup'],'archived_C2_ci':old['speedup_ci'],'reconstructed_C2_score':reconstruct(cs),'reported_score_ratio':result['speedup']/old['speedup'],'prepare_seconds':setup['prepare_s'],'official_tests':len(tests),'profiles':len(pnew),'checks':result['checks'],'flags':result['flags'],'coverage':dict(zip(('overrides','fallbacks','pick_keys','math_checks','captures','fresh'),map(int,coverage.groups()))),'target_raw_regressions':sum(r['new_raw_us']>r['old_raw_us'] for r in target),'target_raw_change_percent_range':[min(100*(r['new_raw_us']/r['old_raw_us']-1) for r in target),max(100*(r['new_raw_us']/r['old_raw_us']-1) for r in target)],'target_regressions':sum(r['delta_us']>0 for r in target),'target_change_percent_range':[min(r['latency_change_percent'] for r in target),max(r['latency_change_percent'] for r in target)],'untargeted_faster':sum(r['delta_us']<0 for r in other),'untargeted_slower':sum(r['delta_us']>0 for r in other),'counterfactual_replace10_score_with_old_baselines':reconstruct(hybrid),'counterfactual_replace10_factor_vs_old':reconstruct(hybrid)/old['speedup'],'counterfactual_scope':'Only ten new adjusted medians inserted into old C2 other20 and old baseline medians. Different seeds/weights/setup/baselines/overhead/tuner/environment. Not paired, not a candidate official score or causal attribution.','machine':result.get('machine'),'control_machine':old.get('machine'),'closure':'Close exact BEFORE-A/global-record SX production variant; no public retry, layer port or unsupported attribution.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'rows30.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'loads6.json').write_text(json.dumps(loads,indent=2)+'\n');(out/'resources.json').write_text(json.dumps(resource,indent=2)+'\n')
 fields=[k for k in rows[0] if k not in ('new_profile','old_profile')]
 with (out/'rows30.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
 print(json.dumps(summary,indent=2));print('TARGETS');print('\n'.join(f"{x['layer']} {x['m']}: {x['old_adjusted_us']:.6f} -> {x['new_adjusted_us']:.6f} ({x['latency_change_percent']:+.3f}%)" for x in target))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.root,a.output)
