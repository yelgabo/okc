"""Static sm90a comparison; no GPU execution or timing."""
import ast,json,argparse,hashlib
from pathlib import Path
from proj_sparse_xor_analyze import entries,find,resources

def analyze(artifact,compiled,prior):
 s=artifact.read_text(encoding='utf8');configs=None
 for n in ast.parse(s).body:
  if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_XOR_CONFIG':configs=ast.literal_eval(n.value)
 e=entries(compiled/'device.sass');old=entries(prior/'device.sass');rows=[]
 for name,c in configs.items():
  args=c['original_targs'].split(', ');row=dict(name=name,variants=[])
  for nearby in (False,True):
   for v in (0,1,2):
    a=args.copy();a[9]=str((int(a[9])&~32 if nearby else int(a[9]))|(4096 if v else 0)|(8192 if v==2 else 0))
    sym,x=find(e,a);r=resources(compiled/'ptxas.log',sym)
    if v<2:
     _,p=find(old,a);assert x['ops']==p['ops'],(name,v,nearby,'control drift')
    assert r['stack']==r['spill_stores']==r['spill_loads']==0,(name,v,nearby,r)
    row['variants'].append(dict(variant=v,nearby=nearby,**r,hmma=x['hmma'],instructions=x['instructions'],global_u16=x['load_u16'],shared_u16=sum('LDS.U16' in z for z in x['ops']),opcode_operand_sha256=hashlib.sha256('\n'.join(x['ops']).encode()).hexdigest(),control_unchanged=(True if v<2 else None)))
  rows.append(row)
 return rows

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('compiled',type=Path);p.add_argument('prior',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 rows=analyze(a.artifact,a.compiled,a.prior);a.output.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
 print([(r['name'],[(v['variant'],v['nearby'],v['regs'],v['instructions'],v['global_u16'],v['shared_u16']) for v in r['variants']]) for r in rows])
