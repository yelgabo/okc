"""Owning stage-local receipt analysis; archived-C2 comparisons are unpaired."""
import argparse,ast,csv,hashlib,json,math,re
from pathlib import Path

CONTROL='9eba96894ee1402e'
SOURCE='de1111b98072f5251ff9b2edd5d453cc22eb8fe1710a943c60922909cc796229'
LAYERS=('qkv_gate','o','pooler','unpooler','router');MS=(1,8,16,32,128,256)
TARGETS={('qkv_gate',8):'qkv8F',('qkv_gate',16):'qkv16F',('qkv_gate',32):'qkv32',('o',16):'o16F',('o',32):'o32F',('pooler',16):'pool16F',('pooler',32):'pool32F',('unpooler',8):'slice8F',('unpooler',16):'unp16F',('unpooler',32):'unp32F'}
FIELDS=['registers','local_bytes','static_shared_bytes','dynamic_shared_bytes','threads','max_threads','resident_blocks_per_SM_admission']

def key(row):return row['case']['layer'],row['case']['m']

def benchmarks(result):
 rows=[r for r in result['records'] if r.get('event')=='benchmark']
 assert len(rows)==30 and len({key(r) for r in rows})==30,'THIRTY_UNIQUE_CELLS'
 assert {key(r) for r in rows}=={(l,m) for l in LAYERS for m in MS},'CASE_COVERAGE'
 for r in rows:
  assert r['status']=='pass','BENCHMARK_NOT_PASS'
  assert r['case']['weight']==62,'OFFICIAL_WEIGHT'
  for f in ('median_ns','median_raw_ns','baseline_ns','baseline_raw_ns'):assert math.isfinite(r[f]) and r[f]>0,('FINITE_POSITIVE',key(r),f)
 return {key(r):r for r in rows}

def score(by,field='median_ns',baseline_field='baseline_ns'):
 return math.prod(sum(by[l,m][baseline_field] for l in LAYERS)/sum(by[l,m][field] for l in LAYERS) for m in MS)**(1/6)

def profiles(result,by):
 rows=[r for r in result['records'] if r.get('event')=='profile']
 assert len(rows)==30 and len({key(r) for r in rows})==30,'THIRTY_UNIQUE_PROFILES'
 assert {key(r) for r in rows}==set(by),'PROFILE_COVERAGE'
 return {key(r):r['report'] for r in rows}

def layout_functions(source):
 ns={};tree=ast.parse(source)
 for n in tree.body:
  if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_XOR_CONFIG':ns['_XOR_CONFIG']=ast.literal_eval(n.value)
 funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('xor_expected_layout','sxs_expected_layout')]
 assert len(funcs)==2 and len(ns['_XOR_CONFIG'])==10,'SOURCE_LAYOUT_HELPERS'
 exec(compile(ast.Module(funcs,[]),'actual-pure-source-layout','exec'),ns)
 return ns

def parse_qualification(log,source):
 ns=layout_functions(source);routes={}
 pattern=r'^SXSRELEASE_ROUTE (\w+) (\d+) (override|unsupported_pack_original_C2) checks=(\d+) capacity=(\d+) arena_bytes=(\d+) resources=(\[.*\])$'
 for line in log.splitlines():
  if not line.startswith('SXSRELEASE_ROUTE '):continue
  match=re.fullmatch(pattern,line);assert match,('ROUTE_SYNTAX',line)
  layer,m,status,checks,capacity,arena=match.groups()[:6];k=(layer,int(m));assert k in TARGETS and k not in routes,'ROUTE_DUPLICATE_OR_UNKNOWN'
  vectors=ast.literal_eval(match[7]);checks=int(checks);capacity=int(capacity);arena=int(arena)
  route=dict(layer=layer,m=int(m),status=status,checks=checks,stage_capacity=capacity,stage_record_bytes=arena,order=['C2_full','stage_full','C2_nf','stage_nf'],fields=FIELDS,vectors=vectors)
  if status=='override':
   assert checks==16 and len(vectors)==4 and all(len(v)==7 for v in vectors),'RESOURCE_COVERAGE'
   expected=ns['sxs_expected_layout'](TARGETS[k],2,capacity)
   assert arena==expected['record_bytes'],'CAPACITY_ARENA'
   assert 0<=expected['dynamic']<=227*1024,'DYNAMIC_EXTENT'
   control_pmax=[]
   for i,v in enumerate(vectors):
    assert all(isinstance(x,int) and x>=0 for x in v),'RESOURCE_VALUES'
    assert v[1]==0 and v[6]>=1,'LOCAL_OR_ADMISSION'
    if i%2:
     assert v[2]==expected['static'] and v[3]==expected['dynamic'] and v[4]==v[5]==expected['threads'],'STAGE_LAYOUT'
     assert v[6]>=vectors[i-1][6],'C2_ADMISSION_REGRESSION'
    else:
     control=ns['sxs_expected_layout'](TARGETS[k],0,0)
     delta=v[3]-control['dynamic'];assert delta>=16 and delta%16==0,'CONTROL_PMAX_LAYOUT'
     control_pmax.append(delta//8)
     assert v[2]==control['static'] and v[4]==v[5]==control['threads'],'CONTROL_LAYOUT'
   assert len(set(control_pmax))==1,'CONTROL_FULL_NF_PMAX_MISMATCH'
   route['inferred_C2_pmax']=control_pmax[0]
   route['stage_expected_layout']=expected
  else:assert checks==capacity==arena==0 and vectors==[],'FALLBACK_RESOURCE_CLAIM'
  routes[k]=route
 assert set(routes)==set(TARGETS),'TEN_ROUTE_COVERAGE'
 matches=re.findall(r'^SXSRELEASE_DONE overrides=(\d+) fallbacks=(\d+) pick_keys=(\d+) checks=(\d+) captures=(\d+) fresh=(\d+)$',log,re.M)
 assert len(matches)==1,'DONE_COVERAGE'
 coverage=dict(zip(('overrides','fallbacks','pick_keys','math_checks','captures','fresh'),map(int,matches[0])))
 active={k for k,v in routes.items() if v['status']=='override'};n=len(active)
 keys=sum(m-(2 if m==8 else 9 if m==16 else 17)+1 for _,m in active)
 assert list(coverage.values())==[n,10-n,keys,n*16,n,n*3],'QUALIFICATION_TOTALS'
 return routes,coverage

def analyze_records(receipt,control,log,source,job):
 assert receipt['status']==control['status']=='done','TERMINAL_STATUS'
 assert (receipt['id'],receipt['board'],receipt['mode'])==(job,'muse_bf16_proj','run'),'OWNING_RECEIPT'
 assert (control['id'],control['board'],control['mode'])==(CONTROL,'muse_bf16_proj','run'),'CONTROL_RECEIPT'
 result=receipt['result'];old=control['result']
 assert result['passed'] and result['checks']['ok'] and not result['flags'],'TERMINAL_NOT_CLEAN_PASS'
 assert old['passed'] and old['checks']['ok'] and not old['flags'],'CONTROL_NOT_CLEAN_PASS'
 by=benchmarks(result);cb=benchmarks(old)
 assert math.isclose(score(by),result['speedup'],rel_tol=1e-13),'SCORE_RECONSTRUCTION'
 assert math.isclose(score(cb),old['speedup'],rel_tol=1e-13),'CONTROL_SCORE_RECONSTRUCTION'
 setup=[r for r in result['records'] if r.get('event')=='setup'];assert len(setup)==1 and math.isfinite(setup[0]['prepare_s']) and 0<=setup[0]['prepare_s']<300,'SETUP_RECORD'
 tests=[r for r in result['records'] if r.get('event')=='test'];assert len(tests)==9 and all(r['status']=='pass' for r in tests),'OFFICIAL_TESTS'
 newprofiles=profiles(result,by);oldprofiles=profiles(old,cb)
 routes,coverage=parse_qualification(log,source)
 rows=[];loads=[];hybrid={}
 for k,b in cb.items():
  hybrid[k]=dict(b,median_ns=by[k]['median_ns'] if k in TARGETS else b['median_ns'])
 for m in MS:
  for l in LAYERS:
   k=l,m;n,b=by[k],cb[k];rows.append(dict(layer=l,m=m,target=k in TARGETS,stage_override=k in routes and routes[k]['status']=='override',new_seed=n['case']['seed'],old_seed=b['case']['seed'],new_adjusted_us=n['median_ns']/1000,old_adjusted_us=b['median_ns']/1000,delta_us=(n['median_ns']-b['median_ns'])/1000,latency_change_percent=100*(n['median_ns']/b['median_ns']-1),new_raw_us=n['median_raw_ns']/1000,old_raw_us=b['median_raw_ns']/1000,new_offset_us=(n['median_raw_ns']-n['median_ns'])/1000,old_offset_us=(b['median_raw_ns']-b['median_ns'])/1000,new_baseline_us=n['baseline_ns']/1000,old_baseline_us=b['baseline_ns']/1000,new_baseline_raw_us=n['baseline_raw_ns']/1000,old_baseline_raw_us=b['baseline_raw_ns']/1000,new_profile=newprofiles[k],old_profile=oldprofiles.get(k)))
  nsum=sum(by[l,m]['median_ns'] for l in LAYERS);bsum=sum(cb[l,m]['median_ns'] for l in LAYERS);hsum=sum(hybrid[l,m]['median_ns'] for l in LAYERS)
  loads.append(dict(m=m,candidate_sum_us=nsum/1000,archived_C2_sum_us=bsum/1000,candidate_baseline_sum_us=sum(by[l,m]['baseline_ns'] for l in LAYERS)/1000,archived_baseline_sum_us=sum(cb[l,m]['baseline_ns'] for l in LAYERS)/1000,counterfactual_replace10_sum_us=hsum/1000,counterfactual_factor_vs_archived_load=bsum/hsum))
 target=[r for r in rows if r['target']]
 summary=dict(job=job,control_job=CONTROL,official_score=result['speedup'],official_ci=result['speedup_ci'],reconstructed_score=score(by),archived_C2_official_score=old['speedup'],archived_C2_ci=old['speedup_ci'],reconstructed_C2_score=score(cb),reported_score_ratio=result['speedup']/old['speedup'],raw_sixload_ratio_using_raw_baselines=score(by,'median_raw_ns','baseline_raw_ns'),archived_raw_sixload_ratio_using_raw_baselines=score(cb,'median_raw_ns','baseline_raw_ns'),prepare_seconds=setup[0]['prepare_s'],official_tests=len(tests),profiles=len(newprofiles),coverage=coverage,target_regressions=sum(r['delta_us']>0 for r in target),target_raw_regressions=sum(r['new_raw_us']>r['old_raw_us'] for r in target),target_change_percent_range=[min(r['latency_change_percent'] for r in target),max(r['latency_change_percent'] for r in target)],counterfactual_replace10_score_with_old_baselines=score(hybrid),counterfactual_replace10_factor_vs_old=score(hybrid)/old['speedup'],comparison_scope='Across-job archived C2 comparison: different input seeds, baseline measurements, environment, tuner and overhead. Ten-cell replacement is a sensitivity scenario, not a paired result or official hybrid score. Recovery from failed global SX is not evidence of a win over C2.',machine=result.get('machine'),control_machine=old.get('machine'),checks=result['checks'],flags=result['flags'])
 return summary,rows,loads,{l+':'+str(m):v for (l,m),v in routes.items()}

def analyze(receipt,control,log,source,output,job):
 raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE,'SOURCE_PIN'
 r=json.loads(receipt.read_text(encoding='utf8'));c=json.loads(control.read_text(encoding='utf8'))
 summary,rows,loads,resources=analyze_records(r,c,log.read_text(encoding='utf8'),raw.decode('utf8'),job)
 summary.update(source_sha256=SOURCE,source_bytes=len(raw),receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),control_receipt_sha256=hashlib.sha256(control.read_bytes()).hexdigest())
 output.mkdir(parents=True,exist_ok=True)
 for name,value in [('summary',summary),('rows30',rows),('loads6',loads),('resources',resources)]:
  (output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8',newline='\n')
 fields=[k for k in rows[0] if k not in ('new_profile','old_profile')]
 with (output/'rows30.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--job',required=True);a=p.parse_args()
 print(json.dumps(analyze(a.receipt,a.control,a.log,a.source,a.output,a.job),indent=2))
