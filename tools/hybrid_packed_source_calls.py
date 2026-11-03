"""CPU-only real module import/native source-assembly capture. Never compiles or launches CUDA."""
import argparse,ast,contextlib,hashlib,importlib.util,json,sys,types,re,tokenize,io
from pathlib import Path
from unittest.mock import patch
import torch
import torch.utils.cpp_extension as cpp

class Symbol:
 def __call__(self,*a,**kw):
  if len(a)==1 and callable(a[0]) and not kw:return a[0]
  return self
 def __getattr__(self,k):return self
 def __getitem__(self,k):return self
class FakeModule(types.ModuleType):
 def __getattr__(self,k):
  if k.startswith('__'):raise AttributeError(k)
  if k in ('jit','autotune','heuristics'):
   return lambda *a,**kw:a[0] if len(a)==1 and callable(a[0]) and not kw else lambda f:f
  return Symbol()
def missing_imports(sources):
 modules={}
 def add(name):
  if name in modules:return modules[name]
  m=FakeModule(name);m.__path__=[];modules[name]=m
  if '.' in name:
   parent,child=name.rsplit('.',1);setattr(add(parent),child,m)
  return m
 for source in sources:
  for n in ast.walk(ast.parse(source)):
   if isinstance(n,ast.Import):
    for alias in n.names:
     if alias.name.startswith(('triton','sglang','sgl_kernel','tvm_ffi')):add(alias.name)
   if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith(('triton','sglang','sgl_kernel','tvm_ffi')):
    mod=add(n.module)
    for a in n.names:
     name=n.module+'.'+a.name
     if a.name in ('gluon','language','scalar_types'):setattr(mod,a.name,add(name))
 return modules
class Ext(types.ModuleType):
 def __getattr__(self,k):
  if k.startswith('__'):raise AttributeError(k)
  return lambda *a,**kw:None

def capture(source,family,mutate=None):
 calls=[]
 def load_inline(*a,**kw):
  assert len(a)<=2,'unexpected positional builder ABI'
  for key,value in zip(('name','cpp_sources'),a):assert key not in kw;kw[key]=value
  calls.append(kw);return Ext(kw['name'])
 modules=missing_imports([source]);
 if 'tvm_ffi.cpp' in modules:modules['tvm_ffi.cpp'].load_inline=load_inline
 ns={'__name__':'_packed_'+family,'__file__':'<packed-'+family+'>'}
 with patch.dict(sys.modules,modules),patch.object(cpp,'load_inline',load_inline),patch.object(torch.cuda,'get_device_capability',return_value=(9,0)):
  exec(compile(source,ns['__file__'],'exec'),ns)
  if mutate is not None:mutate(ns)
  if family=='owned':
   for name in ('_moe1_load','_one_load','_ext','_dlplt_build'):
    ns[name]()
   if '_SKP_WD_SRC' in ns:ns['_skp_load']()
   assert ns['_moe1_source']().count('case 10: launch_main<51>')==1,'missing live MOE51 host dispatch'
  else:
   for name in ('_att7','bfp__get_ext','_graft_glue_module','_graft_b2_module','_sl_cluster_module','_de_ext','_router_fp32_ext','_gle_ext'):ns[name]()
   # Execute the actual per-M selector with exact native-eligible prepared flags.
   state={'ma_sfx':ns['_MA_SFX_IF_PRMT'],'ma_sc6':True,'ma_ext':{}}
   for m in (1,8,16,32,128,256):ns['_fork_native_moe'](m,state)
   assert set(state['ma_ext'])=={'tiny','tiny2','mid','b4','main'},state['ma_ext'].keys()
 return calls,ns

def uncomment(source,spans):
 for a,z in reversed(spans(source)):source=source[:a]+' '+''.join(c for c in source[a:z] if c in '\r\n')+source[z:]
 return source

def canonical(source,spans):
 source=uncomment(source,spans)
 # These are only source-hash-derived C++ symbol renamings performed by the
 # actual _de_ext/_gle_ext builders; no other operands/flags are normalized.
 return re.sub(r'\b(sk2_kernel_narrow_|launch_sk2_narrow_|sk2_kernel_gle|launch_sk2_gle|sk2_gle|sk2f_gle)[a-f0-9]{12}\b',r'\1SOURCEHASH',source)

def check(directory,owner):
 sp=importlib.util.spec_from_file_location('_packed_lexer',owner);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
 report=[]
 packages=[]
 for mask in (4,15):
  raw=(directory/f'hybrid-mask{mask}.py').read_bytes();tree=ast.parse(raw.decode())
  values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)}
  assert values['_VERIFY_OVERRIDE_MASK']==mask
  for family,stem in [('G','G'),('OWNED',f'owned-{mask}')]:assert values['_VERIFY_HYBRID_SOURCE_'+family]==(directory/(stem+'-comment.py')).read_text(encoding='utf8')
  packages.append(dict(mask=mask,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),embedded_backend_identity=True))
 for family,stem in [('G','G'),('owned','owned-4'),('owned','owned-15')]:
  original=(directory/(stem+'-original.py')).read_text(encoding='utf8');revised=(directory/(stem+'-comment.py')).read_text(encoding='utf8')
  xt,yt=ast.parse(original),ast.parse(revised)
  selected=lambda t:[ast.dump(n) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom))]
  assert selected(xt)==selected(yt),'Python definitions/imports drift'
  starts=[0]
  for line in original.splitlines(keepends=True):starts.append(starts[-1]+len(line))
  spans=[(starts[t.start[0]-1]+t.start[1],starts[t.end[0]-1]+t.end[1]) for t in tokenize.generate_tokens(io.StringIO(original).readline) if t.type==tokenize.COMMENT]
  protected_py=mod.protected(original,spans)
  for start,end in protected_py:assert original[start:end] in revised,'Python legal/credit comment lost'
  before,ns0=capture(original,family);after,ns1=capture(revised,family)
  assert len(before)==len(after),(stem,len(before),len(after))
  rows=[]
  for i,(a,b) in enumerate(zip(before,after)):
   fields={};raw_before={};raw_after={};legal_count=0
   for field in ('cpp_sources','cuda_sources'):
    def strings(v):return v if isinstance(v,list) else [v] if isinstance(v,str) else []
    aa,bb=strings(a.get(field)),strings(b.get(field));assert len(aa)==len(bb),(stem,i,field)
    for j,(x,y) in enumerate(zip(aa,bb)):
     assert canonical(x,mod.comment_spans)==canonical(y,mod.comment_spans),(stem,i,field,j,'noncomment source assembly drift')
     assert '@@' not in y,'unexpanded source placeholder'
     protected=mod.protected(x,mod.comment_spans(x))
     for start,end in protected:assert x[start:end] in y,'retained legal comment group lost'
     legal_count+=len(protected)
    raw_before[field]=[hashlib.sha256(x.encode()).hexdigest() for x in aa];raw_after[field]=[hashlib.sha256(y.encode()).hexdigest() for y in bb]
    fields[field]=[hashlib.sha256(canonical(y,mod.comment_spans).encode()).hexdigest() for y in bb]
   for key in set(a)|set(b):
    if key not in ('name','cpp_sources','cuda_sources'):assert a.get(key)==b.get(key),(stem,i,key,a.get(key),b.get(key))
   rows.append(dict(original_name=a['name'],packed_name=b['name'],functions=b.get('functions'),normalized_source_sha256=fields,raw_original_source_sha256=raw_before,raw_packed_source_sha256=raw_after,protected_legal_comments_checked=legal_count,flags=b.get('extra_cuda_cflags')))
  report.append(dict(family=stem,input_sha256=hashlib.sha256(original.encode()).hexdigest(),packed_sha256=hashlib.sha256(revised.encode()).hexdigest(),real_load_inline_calls=len(rows),identical_python_definitions_imports=len(selected(xt)),python_legal_comments_checked=len(protected_py),captures=rows))
 return dict(scope='Actual full Python backend module imports + source factories executed. Missing SGLang/Triton imports, GPU capability and load_inline are explicit CPU stubs; no native host link/device/runtime proof.',packages=packages,records=report)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('lexer',type=Path);p.add_argument('output',type=Path);a=p.parse_args();report=check(a.directory,a.lexer);a.output.write_text(json.dumps(report,indent=2)+'\n');print([(r['family'],r['real_load_inline_calls']) for r in report['records']])
