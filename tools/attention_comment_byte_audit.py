import sys,re,ast,json,hashlib
from pathlib import Path
sys.path[:0]=['../okc-today-proj/tools','../okc/tools']
import prod_size_comments as p,dl_g_production_slice as gp,dl_owned_verify_production as op
r=Path('../okc');g,_=gp.build((r/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf8'));o,_=op.build(r,(16,32));records=[]
for name,s in [('G',g),('OWNED',o)]:
 out,proof=p.process(s)
 for n in ast.parse(s).body:
  if not(isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) and len(n.value.value)>1000):continue
  v=n.value.value
  if not re.search(r'#include|__device__|__global__|std::vector|namespace ',v):continue
  w,info=p.strip_cpp(v);phase=lambda x:re.sub(r'\\\r?\n','',x)
  # Independent phase2-first scanner comparison: exact whitespace retained except
  # comment-splice newlines; normalize only whitespace after stripping comments.
  def tokens(x):
   x=phase(x);sp=p.comment_spans(x)
   for a,b in reversed(sp):x=x[:a]+' '+x[b:]
   return re.findall(r'R"[^ ()\\\t\r\n]*\(.*?\)[^ ()\\\t\r\n]*"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_$][\w$]*|\d+(?:\.\d*)?|\S',x,re.S)
  assert tokens(v)==tokens(w),(name,ast.unparse(n.targets[0]))
  assert v.count('\n')==w.count('\n')
  records.append(dict(owner=name,name=ast.unparse(n.targets[0]),sha256=hashlib.sha256(v.encode()).hexdigest(),stripped_sha256=hashlib.sha256(w.encode()).hexdigest(),lines=v.count('\n'),phase2_token_comparison=True,**info))
 print(name,'native',sum(x['owner']==name for x in records),'AST check PASS')
pth=Path('docs/steering/evidence/2026-11-01-comment-byte-review');pth.mkdir(parents=True,exist_ok=True);(pth/'native-census.json').write_text(json.dumps(records,indent=2),encoding='utf8')
