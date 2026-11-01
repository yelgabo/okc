"""Required fractional latency cuts from owned paired controls; no speed forecast."""
import hashlib,json,math,statistics
from pathlib import Path

def analyze(path):
 raw=path.read_bytes();data=json.loads(raw);report=data['report'];offsets={tuple(r[:3]):r[3] for r in data['contemporary_offsets']}
 cells={}
 for b,l,w,seed,times in report['rows']:
  key=(b,l,w);cells.setdefault(key,[]).append(statistics.mean(times[:2])-offsets[key])
 med={k:statistics.mean(v) for k,v in cells.items()};assert len(med)==12 and min(med.values())>0
 def gain(cut,targets):
  logs=[]
  for b in (1,16,32):
   for l in (1,8):
    old=sum((46 if w else 16)*med[b,l,w] for w in (0,1))
    new=sum((46 if w else 16)*med[b,l,w]*(1-cut if (b,l,w) in targets else 1) for w in (0,1))
    logs.append(math.log(old/new))
  return math.exp(sum(logs)/6)
 sets={'verify16_32_full':{(16,8,0),(32,8,0)},'all16_32_full':{(b,l,0) for b in (16,32) for l in (1,8)}}
 required={}
 for name,targets in sets.items():
  results={}
  for goal in (1.01,2.7119/2.6043):
   lo,hi=0.,.999
   for _ in range(64):
    mid=(lo+hi)/2
    if gain(mid,targets)<goal:lo=mid
    else:hi=mid
   results[str(goal)]={'fractional_cut_required':hi,'cell_us_savings':{str(k):med[k]*hi for k in sorted(targets)}}
  required[name]=results
 return {'receipt_job':data['job'],'input_sha256':hashlib.sha256(raw).hexdigest(),'control_adjusted_us':{str(k):v for k,v in sorted(med.items())},'required_sensitivity':required,'meaning':'Common multiplicative latency-cut sensitivity on paired owned control means, all other cells fixed; not measured candidate gain or theoretical attainable bound.'}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('analysis',type=Path);p.add_argument('output',type=Path);a=p.parse_args();r=analyze(a.analysis);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf8');print(json.dumps(r,indent=2))
