"""Transparent comment-only byte plan, with protected legal blocks and token proof."""
import argparse,ast,hashlib,importlib.util,io,json,re,sys,tokenize
from pathlib import Path

LEGAL=re.compile(r'copyright|licen[cs]e|\bNOTICE\b|SPDX|attribution|portions|https?://|github|bet4|alygg|vjeux|keell|matvey|Nathen|chen|Zhimao|SGLang|Marlin|Bohdan|Matsvei|rohan|author|credit|adapted|derived|ported|courtesy|claude|gemini|anthropic|OpenAI|@@|//@',re.I)

def _direct_comment_spans(s):
 result=[];i=0
 while i<len(s):
  if s.startswith('//',i):
   z=s.find('\n',i)
   if z<0:z=len(s)
   # Translation phase 2 splices backslash-newline before recognizing comments.
   while z<len(s) and s[i:z].rstrip('\r').endswith('\\'):
    z=s.find('\n',z+1)
    if z<0:z=len(s)
   result.append((i,z));i=z
  elif s.startswith('/*',i):
   z=s.find('*/',i+2);assert z>=0,'unterminated native comment';z+=2;result.append((i,z));i=z
  elif s.startswith('R"',i):
   a=s.find('(',i+2);assert a>=0;delimiter=s[i+2:a];z=s.find(')'+delimiter+'"',a+1);assert z>=0;i=z+len(delimiter)+2
  elif s[i] in ('"',"'"):
   quote=s[i];i+=1
   while i<len(s):
    if s[i]=='\\':i+=2
    elif s[i]==quote:i+=1;break
    else:i+=1
  else:i+=1
 return result

def comment_spans(s):
 # Fail closed on a delimiter created by phase-2 splicing. Actual inputs are
 # independently censused; supporting arbitrary splice-created tokens is not
 # needed to strip their comments and must not be silently approximated.
 logical=[];positions=[];i=0
 while i<len(s):
  if s[i]=='\\' and s[i+1:i+2]=='\n':i+=2;continue
  if s[i]=='\\' and s[i+1:i+3]=='\r\n':i+=3;continue
  logical.append(s[i]);positions.append(i);i+=1
 text=''.join(logical)
 for a,z in _direct_comment_spans(text):
  if positions[a+1]!=positions[a]+1:raise ValueError('unsupported phase2-created comment delimiter')
  if text[a:a+2]=='/*' and positions[z-1]!=positions[z-2]+1:raise ValueError('unsupported phase2-created comment delimiter')
 return _direct_comment_spans(s)

def protected(s,spans):
 groups=[]
 for a,z in spans:
  if groups and not s[groups[-1][-1][1]:a].strip():groups[-1].append((a,z))
  else:groups.append([(a,z)])
 return {(a,z) for group in groups if LEGAL.search('\n'.join(s[a:z] for a,z in group)) for a,z in group}

def strip_cpp(s,anchors=()):
 spans=comment_spans(s);keep=protected(s,spans);legal=set(keep);out=s
 for anchor in anchors:
  p=s.find(anchor)
  while p>=0:
   keep|={(a,z) for a,z in spans if a<p+len(anchor) and z>p}
   p=s.find(anchor,p+1)
 for a,z in spans:
  prefix=re.sub(r'\\\r?\n','',s[:a]).rsplit('\n',1)[-1]
  if '\n' in s[a:z] and prefix.lstrip().startswith('#'):keep.add((a,z))
 for a,z in reversed(spans):
  if (a,z) not in keep:out=out[:a]+' '+''.join(c for c in s[a:z] if c in '\r\n')+out[z:]
 # Compare the exact non-comment lexical character stream with whitespace kept:
 # replacing comments only by whitespace preserves token separation/newlines.
 def code(t):
  spans=comment_spans(t)
  for a,z in reversed(spans):t=t[:a]+' '+''.join(c for c in t[a:z] if c in '\r\n')+t[z:]
  return t
 assert code(s)==code(out),'native non-comment character drift'
 assert all(s[a:z] in out for a,z in keep),'legal native block lost'
 return out,dict(comment_bytes=sum(len(s[a:z].encode()) for a,z in spans),protected_bytes=sum(len(s[a:z].encode()) for a,z in legal),preserved_additional_comments=len(keep-legal),saved_bytes=len(s.encode())-len(out.encode()),comments=len(spans),protected_comments=len(legal))

def offsets(s):
 lines=s.splitlines(keepends=True);starts=[0]
 for l in lines:starts.append(starts[-1]+len(l))
 return lines,starts

def python_comments(s):
 lines,starts=offsets(s);tokens=list(tokenize.generate_tokens(io.StringIO(s).readline));spans=[];string_lines=set()
 for t in tokens:
  if t.type==tokenize.COMMENT:spans.append((starts[t.start[0]-1]+t.start[1],starts[t.end[0]-1]+t.end[1]))
  if t.type==tokenize.STRING:string_lines.update(range(t.start[0],t.end[0]+1))
 keep=protected(s,spans)
 keep|={(a,z) for a,z in spans if s[a:z].startswith('# okc:')}
 result=s
 for a,z in reversed(spans):
  if (a,z) not in keep:result=result[:a]+result[z:]
 # Preserve all literals and every remaining line of code, including line breaks.
 before=ast.parse(s);after=ast.parse(result);assert ast.dump(before)==ast.dump(after),'Python comment AST drift'
 docstrings=[n.body[0] for n in ast.walk(before) if isinstance(n,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.body and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str)]
 return result,dict(comment_bytes=sum(len(s[a:z].encode()) for a,z in spans),protected_bytes=sum(len(s[a:z].encode()) for a,z in keep),saved_bytes=len(s.encode())-len(result.encode()),blank_lines_bytes=sum(len(l.encode()) for i,l in enumerate(lines,1) if i not in string_lines and not l.strip()),docstring_bytes=sum(len(ast.get_source_segment(s,n).encode()) for n in docstrings),docstrings_removed=False,blank_lines_removed=False)

def process(source):
 # Work on complete pre-factor strings; never lex a shared fragment that might
 # begin in the middle of a block comment or C++ raw string.
 tree=ast.parse(source);replacements=[];records={}
 native_values={id(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) and len(n.value.value)>1000 and re.search(r'#include|__device__|__global__|std::vector|namespace ',n.value.value)}
 methods={'replace','split','index','find','count','startswith','endswith'}
 anchors={n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in methods and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str) and ('//' in n.args[0].value or '/*' in n.args[0].value)}
 # Include local named anchors/old strings, not just inline call arguments.
 # Native literal bodies themselves are not anchors unless explicitly used as
 # one; those are separately covered by actual source-assembly capture tests.
 anchors|={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and id(n) not in native_values and ('//' in n.value or '/*' in n.value)}
 for n in tree.body:
  if not (isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) and len(n.value.value)>1000):continue
  value=n.value.value
  if not re.search(r'#include|__device__|__global__|std::vector|namespace ',value):continue
  revised,record=strip_cpp(value,anchors);label=ast.unparse(n.targets[0]);records[label]=record
  # Reuse the existing transparent-literal renderer; string value is reviewable.
  quote=next(q for q in ("'''",'"""') if q not in revised)
  rendered='r'+quote+revised+quote
  assert ast.literal_eval(rendered)==revised
  replacements.append((n.value.lineno,n.value.col_offset,n.value.end_lineno,n.value.end_col_offset,rendered))
 lines,starts=offsets(source);out=source
 for l,c,el,ec,value in sorted(replacements,reverse=True):
  a=starts[l-1]+len(lines[l-1].encode()[:c].decode());z=starts[el-1]+len(lines[el-1].encode()[:ec].decode());out=out[:a]+value+out[z:]
 out,py=python_comments(out)
 return out,dict(native=records,python=py,preserved_runtime_comment_anchors=sorted(anchors),total_saved_before_factoring=len(source.encode())-len(out.encode()),input_sha256=hashlib.sha256(source.encode()).hexdigest(),output_sha256=hashlib.sha256(out.encode()).hexdigest())

def screen(root,out,emit=None):
 root=root.resolve();saved=list(sys.path)
 try:
  sys.path.insert(0,str(root/'tools'))
  import dl_g_production_slice as gp,dl_g_literal_factor as factor,dl_owned_verify_production as owned,dl_verify_hybrid as hybrid
  g,_=gp.build((root/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf8'));g0,_=factor.factor(g)
  gs,gproof=process(g);gs,fp=factor.factor(gs)
  if emit:
   emit.mkdir(parents=True,exist_ok=True)
   (emit/'G-original.py').write_bytes(g0.encode());(emit/'G-comment.py').write_bytes(gs.encode())
  results={};proofs={'G':gproof};sources={}
  for batches,mask in [((32,),4),((16,32),15)]:
   o,_=owned.build(root,batches);os,op=process(o);proofs[str(batches)]=op
   before,bm=hybrid.build(root,mask,g_backend=g0,owned_backend=o);after,am=hybrid.build(root,mask,g_backend=gs,owned_backend=os)
   results[str(batches)]=dict(original_bytes=bm['bytes'],comment_only_bytes=am['bytes'],saved=bm['bytes']-am['bytes'],margin_to_strict_cap=999999-am['bytes'],G_bytes=len(gs.encode()),owned_bytes=len(os.encode()),no_blank_or_docstring_removal=True)
   sources[mask]=after
   if emit:
    for name,value in [('owned-'+str(mask)+'-original.py',o),('owned-'+str(mask)+'-comment.py',os),('hybrid-mask'+str(mask)+'.py',after)]:
     (emit/name).write_bytes(value.encode())
 finally:sys.path[:]=saved
 report=dict(results=results,proofs=proofs,assumption='No constexpr/docstring/blankline removal; author/legal groups and runtime anchors retained. --emit writes readable candidates; source fit is not H200 qualification.',GPU=False)
 report['tool_lf_sha256']=hashlib.sha256(Path(__file__).read_text(encoding='utf8').encode()).hexdigest()
 if emit:
  report['artifacts']={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(emit.glob('*.py'))}
 report['unchanged_generation_inputs']={n:hashlib.sha256((root/'tools'/n).read_text(encoding='utf8').encode()).hexdigest() for n in ('dl_verify_hybrid.py','dl_verify_hybrid_body.py','dl_g_production_slice.py','dl_g_literal_factor.py','dl_owned_verify_production.py','dl_verify_pair_slices.py')}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(results,indent=2))
 return sources,report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--emit',type=Path);a=p.parse_args();screen(a.root,a.output,a.emit)
