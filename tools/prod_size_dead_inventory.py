"""Named constexpr-discarded bodies: inventory only, not a source transform."""
import argparse,ast,hashlib,json,re
from pathlib import Path

def masked(s):
 pattern=r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27'
 return re.sub(pattern,lambda m:''.join('\n' if c=='\n' else ' ' for c in m[0]),s,flags=re.S)

def end(s,start,left,right):
 assert s[start]==left;depth=1;i=start+1
 while depth:
  assert i<len(s);depth+=(s[i]==left)-(s[i]==right);i+=1
 return i

def truth(condition,env,calls):
 s=condition
 for name,value in calls.items():s=re.sub(r'\b'+name+r'\s*\(\s*MODE\s*\)',str(int(value)),s)
 def variable(m):
  n=m[0]
  if n in env:return str(int(env[n]))
  if n in ('true','false'):return str(int(n=='true'))
  raise ValueError(n)
 try:s=re.sub(r'\b[A-Za-z_]\w*\b',variable,s)
 except ValueError:return None
 s=s.replace('&&',' and ').replace('||',' or ');s=re.sub(r'!(?!=)',' not ',s).strip()
 try:
  t=ast.parse(s,mode='eval')
  if any(not isinstance(n,(ast.Expression,ast.Constant,ast.UnaryOp,ast.Not,ast.USub,ast.UAdd,ast.Invert,ast.BoolOp,ast.And,ast.Or,ast.Compare,ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.BinOp,ast.BitAnd,ast.BitOr,ast.BitXor,ast.Add,ast.Sub,ast.Mult,ast.Mod,ast.FloorDiv,ast.LShift,ast.RShift)) for n in ast.walk(t)):return None
  return bool(eval(compile(t,'constexpr-inventory','eval'),{'__builtins__':{}}))
 except (SyntaxError,ValueError,ZeroDivisionError):return None

def inventory(source,env,calls):
 mask=masked(source);chunks=[]
 for m in re.finditer(r'\bif\s+constexpr\s*\(',mask):
  p=mask.index('(',m.start());q=end(mask,p,'(',')');condition=source[p+1:q-1];value=truth(condition,env,calls)
  if value is None:continue
  a=q
  while mask[a].isspace():a+=1
  if mask[a]!='{':continue
  z=end(mask,a,'{','}')
  if value:
   e=re.match(r'\s*else\s*\{',mask[z:])
   if not e:continue
   a=z+e.end()-1;z=end(mask,a,'{','}')
  chunks.append(dict(condition=condition.strip(),discarded='else' if value else 'then',start=a+1,end=z-1,line=source.count('\n',0,a)+1,bytes=len(source[a+1:z-1].encode())))
 # Count nested discarded statements exactly once. Retain readable empty braces
 # as a conservative inventory; named explanatory markers cost extra if built.
 outer=[]
 for c in sorted(chunks,key=lambda c:(c['start'],-c['end'])):
  if any(o['start']<=c['start'] and o['end']>=c['end'] for o in outer):continue
  outer.append(c)
 return dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),source_bytes=len(source.encode()),raw_discardable_body_bytes=sum(c['bytes'] for c in outer),named_chunks=outer,assumptions=env,helper_results_at_MODE51=calls,scope='Byte upper budget only: no source modified or compile/runtime equivalence claimed')

def run(directory,out):
 env=dict(L=8,COOP=1,ONE=1,SPEC=0,SOLO=0,HMRG=0,DETM=0,FIN8=0,QK2=0,PE16=0,STAMPS=0,WAVE=0,XPIPE=0,ONE_DIAG=0,ONE_STAMP=0,ONE_PROBE=0,ONE_XPIPE=0)
 one=inventory((directory/'owned32-_ONE_SRC.cu').read_text(encoding='utf8'),env,{})
 calls={k:False for k in ('is_diag','is_light','pa_diag','pb_diag','skew_st','grp_hoist','pair_st','defer_a','vol_regs','a_only','b_only')}
 calls.update({k:True for k in ('fast_setup','lean_stage','grp_codes','own_a','dyn_b','fast_silu','bar_release_a','fence_ar','dual_chain','final_ss')});calls['mma_max']=2
 moe=inventory((directory/'owned32-_MOE1_SRC.cu').read_text(encoding='utf8'),dict(MODE=51,kDiag=0,kPa=0,kLight=0),calls)
 result=dict(ONE=one,MOE=moe,raw_budget=one['raw_discardable_body_bytes']+moe['raw_discardable_body_bytes'],minimum_single32_saving=72105,minimum_all4_saving=125549,source_changes=False,GPU=False)
 out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ('ONE','MOE')},indent=2));print('ONE',one['raw_discardable_body_bytes'],'MOE',moe['raw_discardable_body_bytes'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.directory,a.output)
