"""Read-only literal inventory and exact common spans; never emits kernel code."""
import ast,difflib,hashlib,json,sys
from pathlib import Path

def strings(source):
 out=[]
 for top in ast.parse(source).body:
  label=getattr(top,'name',None) or ','.join(ast.unparse(t) for t in getattr(top,'targets',[]))
  for n in ast.walk(top):
   if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value.encode())>=1024:
    out.append((f'{label}:{n.lineno}',n.value))
 return out

def audit(root):
 sys.path.insert(0,str(root/'tools'));from dl_verify_pair_slices import build
 arms,manifest=build(root)
 g=(root/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf8')
 sources={'full_G':g,'four_cell_G':arms['G'],'four_cell_owned':arms['owned']}
 inventory={k:[{'name':n,'bytes':len(v.encode()),'sha256':hashlib.sha256(v.encode()).hexdigest()} for n,v in strings(s)] for k,s in sources.items()}
 common=[]
 for gn,gv in strings(g):
  gl=gv.splitlines(keepends=True)
  for on,ov in strings(arms['owned']):
   ol=ov.splitlines(keepends=True)
   spans=[]
   for a,b,size in difflib.SequenceMatcher(None,gl,ol,autojunk=True).get_matching_blocks():
    value=''.join(gl[a:a+size]);n=len(value.encode())
    if n>=256:
     assert value==''.join(ol[b:b+size])
     spans.append({'G_line':a+1,'owned_line':b+1,'bytes':n,'sha256':hashlib.sha256(value.encode()).hexdigest()})
   if spans:common.append({'G':gn,'owned':on,'pair_common_bytes':sum(s['bytes'] for s in spans),'spans':spans})
 intra=[];values=strings(g)
 for index,(an,av) in enumerate(values):
  for bn,bv in values[:index]:
   al=av.splitlines(keepends=True);bl=bv.splitlines(keepends=True)
   spans=[]
   for a,b,size in difflib.SequenceMatcher(None,al,bl,autojunk=True).get_matching_blocks():
    value=''.join(al[a:a+size]);n=len(value.encode())
    if n>=256:spans.append({'A_line':a+1,'B_line':b+1,'bytes':n,'sha256':hashlib.sha256(value.encode()).hexdigest()})
   if spans:intra.append({'A':an,'B':bn,'pair_common_bytes':sum(s['bytes'] for s in spans),'spans':spans})
 return {'source_pins':{k:hashlib.sha256(s.encode()).hexdigest() for k,s in sources.items()},'source_bytes':{k:len(s.encode()) for k,s in sources.items()},'manifest':manifest,'literal_inventory':inventory,'exact_common_spans':sorted(common,key=lambda x:-x['pair_common_bytes']),'G_internal_common_spans':sorted(intra,key=lambda x:-x['pair_common_bytes']),'warning':'Pair sums overlap; not an achievable byte saving, source pruner or correctness proof. No foreign/device source emitted.'}
if __name__=='__main__':
 root,out=map(Path,sys.argv[1:]);r=audit(root);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf8');print(json.dumps({'source_bytes':r['source_bytes'],'largest_pairs':[{k:v for k,v in x.items() if k!='spans'} for x in r['exact_common_spans'][:8]],'G_internal_pairs':[{k:v for k,v in x.items() if k!='spans'} for x in r['G_internal_common_spans'][:4]]},indent=2))
