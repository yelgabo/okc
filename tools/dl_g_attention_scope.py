"""Offline G/owned attention full-context scope and six-load sensitivities."""
import ast,hashlib,json,math,re,argparse
from pathlib import Path
from dl_proj_budget_audit import counterfactual,load_times

def src_value(path,name):
 text=path.read_text(encoding='utf-8');node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name)
 return ast.literal_eval(node.value)
def profile_time(d,index,needle):
 report=d['profiles'][index]['report'];matches=[]
 for line in report.splitlines():
  m=re.match(r'(.+?)\s+([0-9.]+) µs\s+[0-9.]+%\s+(\d+)$',line)
  if m and needle in m[1]:matches.append(float(m[2]))
 assert len(matches)==1,(index,needle,matches)
 return matches[0]
def load(path):
 d=json.loads(path.read_text(encoding='utf-8'))
 if 'result' in d:
  r=d['result'];return dict(benchmarks=[x for x in r['records'] if x['event']=='benchmark'],profiles=[x for x in r['records'] if x['event']=='profile'],summary={'speedup':r['speedup']})
 return d

def execute_plans(root):
 def function(path,name,env):
  t=ast.parse(path.read_text(encoding='utf-8'));n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
  exec(compile(ast.Module(body=[n],type_ignores=[]),'<actual-shape-plan>','exec'),env)
  return env[name]
 gp=function(root/'muse_decode_layer/cand/g1b4_stage_release.py','_plan7',dict(_num_sms7=lambda _:132,OKC_F2=0))
 op=function(root/'muse_attention_decode/submission.py','_one_plan',dict(_ONE_FLAT=True,_ONE_Q8_PROD=True,_ONE_PROD=True,_ONE_TARGET=264))
 out=[]
 for b,l in ((16,1),(32,1),(16,8),(32,8)):
  g=gp(b,1,l,0,86008,1,'cuda')
  owned=op(dict(batch=b,q_len=l,page_size=1,window=0),86008,{(l,49 if l==1 else 51):264})
  assert g==dict(n_splits=264,flat=1,cs=0,ncl=1,nc=1)
  assert owned==(264,1,49 if l==1 else 51)
  out.append(dict(batch=b,length=l,G=g,owned=owned))
 return out

def audit(root):
 g=load(root/'muse_decode_layer/profiles/g1-stage-release.607eb52bdaf24160.json')
 owned=load(root/'muse_attention_decode/profiles/scored.8914751e3b9f468b.json')
 current=load(root/'muse_attention_decode/cand/profiles/late_full.d331aca81c554d70.json')
 sg=load(root/'muse_attention_decode/baseline_profile/profiles/baseline.24daba2202e547d6.json')
 rows=[];savings={};fullfloor={};gcounters={};ten={}
 for i in (3,5,9,11):
  b=owned['benchmarks'][i];c=g['benchmarks'][i]['case'];got=profile_time(g,i,'::attn_kernel<');old=profile_time(owned,i,'::attn_one_kernel<');now=profile_time(current,i,'::attn_one_kernel<')
  key=(c['load'],c['layer']);savings[key]=max(0,got-old);ten[key]=got*.1
  fullfloor[key]=b['median_ns']/1000-b['speed_of_light_ns']/1000
  rows.append(dict(load=c['load'],batch=c['batch'],length=c.get('q_len',1),weight=c['weight'],G_prof_us=got,owned891_prof_us=old,owned_current_check_prof_us=now,owned891_us=b['median_ns']/1000,owned_current_check_us=current['benchmarks'][i]['median_ns']/1000,standalone_floor_us=b['speed_of_light_ns']/1000,sglang_profile_total_us=sg['benchmarks'][i]['median_ns']/1000,G_dist=c['dist'],owned_dist=b['case']['dist'],standalone_floor_factor=counterfactual(owned,{(b['case']['load'],'full'):fullfloor[key]})))
 gcounters['G_replace_profile_gaps_with891']=counterfactual(g,savings)
 gcounters['G_ten_percent_each4_attention_profile_rows']=counterfactual(g,ten)
 gcounters['standalone_all4_to_official_floor']=counterfactual(owned,fullfloor)
 gcounters['standalone_current_check_all4_to_891_medians']=counterfactual(current,{(b['case']['load'],b['case']['layer']):max(0,b['median_ns']/1000-owned['benchmarks'][i]['median_ns']/1000) for i,b in enumerate(current['benchmarks']) if i in (3,5,9,11)})
 values={}
 for file,name,label in ((root/'muse_decode_layer/cand/g1b4_stage_release.py','ATT7_CUDA_SRC','G'),(root/'muse_attention_decode/submission.py','_ONE_SRC','owned_current')):
  s=src_value(file,name);values[label]={'device_sha256':hashlib.sha256(s.encode()).hexdigest()}
 return dict(rows=rows,sensitivities=gcounters,sources=values,executed_shape_plans=execute_plans(root))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(audit(a.root),indent=2))
