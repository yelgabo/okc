"""Transparent exact-string factoring; visible C++ chunks, no codec or runtime I/O."""
import ast,difflib,hashlib,json
from pathlib import Path

def literal(s):
 for quote in ("'''",'"""'):
  if quote in s:continue
  value='r'+quote+s+quote
  try:
   if ast.literal_eval(value)==s:return value
  except SyntaxError:pass
 raise ValueError('chunk cannot be represented as a visible raw multiline literal')

def assignment(tree,target):
 found=[n for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and ast.unparse(n.targets[0])==target]
 assert len(found)==1,(target,len(found));n=found[0]
 assert isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)
 return n

def pair(source,left,right,prefix,min_bytes=256):
 tree=ast.parse(source);a=assignment(tree,left);b=assignment(tree,right)
 av,bv=a.value.value,b.value.value;al=av.splitlines(keepends=True);bl=bv.splitlines(keepends=True)
 blocks=[]
 for ai,bi,n in difflib.SequenceMatcher(None,al,bl,autojunk=True).get_matching_blocks():
  text=''.join(al[ai:ai+n])
  if len(text.encode())>=min_bytes:blocks.append((ai,bi,n,text))
 assert blocks and not any(isinstance(n,ast.Name) and n.id.startswith(prefix) for n in ast.walk(tree))
 definitions='';names=[]
 for i,(_,_,_,text) in enumerate(blocks):
  name=prefix+str(i);names.append(name);definitions+=name+' = '+literal(text)+'\n\n'
 def expression(lines,side):
  terms=[];cursor=0
  for i,block in enumerate(blocks):
   start=block[side];n=block[2]
   if start>cursor:terms.append(literal(''.join(lines[cursor:start])))
   terms.append(names[i]);cursor=start+n
  if cursor<len(lines):terms.append(literal(''.join(lines[cursor:])))
  return '(\n'+ '\n + '.join(terms)+'\n)'
 replacements=[(a.lineno,a.end_lineno,left+' = '+expression(al,0)+'\n'),(b.lineno,b.end_lineno,right+' = '+expression(bl,1)+'\n')]
 lines=source.splitlines(keepends=True)
 for start,end,text in sorted(replacements,reverse=True):lines[start-1:end]=[text]
 out=''.join(lines);where=min(a.lineno,b.lineno)-1
 # The earliest original assignment's line position is unchanged by later replacements.
 lines=out.splitlines(keepends=True);lines[where:where]=[definitions];out=''.join(lines)
 revised=ast.parse(out);env={};actual={}
 for n in revised.body:
  if not isinstance(n,ast.Assign) or len(n.targets)!=1:continue
  target=ast.unparse(n.targets[0])
  if isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith(prefix):env[target]=ast.literal_eval(n.value)
  elif target in (left,right):actual[target]=eval(compile(ast.Expression(n.value),'<visible-concat>','eval'),{'__builtins__':{}},env)
 assert actual=={left:av,right:bv}
 # Prove every other AST node unchanged; restore only the two original values.
 normalized=[]
 for n in revised.body:
  if isinstance(n,ast.Assign) and len(n.targets)==1:
   target=ast.unparse(n.targets[0])
   if target.startswith(prefix):continue
   if target==left:n.value=ast.Constant(av)
   if target==right:n.value=ast.Constant(bv)
  normalized.append(n)
 assert ast.dump(ast.Module(body=normalized,type_ignores=[]))==ast.dump(tree)
 return out,{'targets':[left,right],'shared_chunks':len(blocks),'common_bytes':sum(len(v[3].encode()) for v in blocks),'saved_source_bytes':len(source.encode())-len(out.encode()),'runtime_sha256':{left:hashlib.sha256(av.encode()).hexdigest(),right:hashlib.sha256(bv.encode()).hexdigest()},'outside_AST_unchanged':True}

def factor(source):
 records=[]
 for left,right,prefix in (('MOE_CUDA_SRC',"_MA_ALT_SRCS['moe_helper0']",'_OKC_VISIBLE_MOE_'),('_GRAFT_GLUE_CUDA','_SL_GLUE_CUDA','_OKC_VISIBLE_GLUE_')):
  source,record=pair(source,left,right,prefix);records.append(record)
 return source,records

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 src=a.source.read_text(encoding='utf8');out,records=factor(src);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(out.encode())
 result={'input_bytes':len(src.encode()),'input_sha256':hashlib.sha256(src.encode()).hexdigest(),'output_bytes':len(out.encode()),'output_sha256':hashlib.sha256(out.encode()).hexdigest(),'records':records,'gpu_execution':False,'readable_multiline_chunks':True}
 a.output.with_suffix('.factor.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
