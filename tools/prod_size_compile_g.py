"""Selected Family-G device-only control/comment compiler gate; no GPU."""
import argparse,ast,hashlib,json,re,subprocess,sys,types
from pathlib import Path

def bindings(source):
 env={}
 def value(n):
  if isinstance(n,ast.Constant):return n.value
  if isinstance(n,ast.Name):return env[n.id]
  if isinstance(n,(ast.Tuple,ast.List)):return tuple(value(x) for x in n.elts)
  if isinstance(n,ast.Dict):return {value(k):value(v) for k,v in zip(n.keys,n.values)}
  if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Add):return value(n.left)+value(n.right)
  if isinstance(n,ast.Subscript):return value(n.value)[value(n.slice)]
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='replace':return value(n.func.value).replace(*(value(x) for x in n.args))
  raise ValueError('not a pure source binding')
 for n in ast.parse(source).body:
  if isinstance(n,ast.Assign) and len(n.targets)==1:
   try:v=value(n.value)
   except (KeyError,ValueError,TypeError,AttributeError):continue
   t=n.targets[0]
   if isinstance(t,ast.Name):env[t.id]=v
   elif isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id in env:
    try:env[t.value.id][value(t.slice)]=v
    except (KeyError,ValueError,TypeError):pass
 return env

def sections(path):
 result={}
 for text in re.split(r'(?=\.section\s+\.text\.)',path.read_text()):
  if not text.startswith('.section'):continue
  name=text.split('.text.',1)[1].split(',',1)[0].strip();ops=[];labels={}
  for line in text.splitlines():
   label=re.match(r'\s*(\.L_\w+):',line)
   if label:labels[label[1]]=len(ops)
   m=re.search(r'/\*[0-9a-f]+\*/\s+([^\n]+?;)',line)
   if m:ops.append(m[1].strip())
  for label,index in sorted(labels.items(),key=lambda x:-len(x[0])):ops=[x.replace(label,'AT_INSTRUCTION_'+str(index)) for x in ops]
  result[name]=[x.replace(name,'SELECTED_SELF') for x in ops]
 return result

def run(root,frozen,out):
 driver=(root/'tools/local_cuda128_compile.py').read_text(encoding='utf8')
 # Preserve the established device boundary/header/declaration adapter and
 # expression list; replace only its hardcoded MAIN flags by per-build flags.
 old="'-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1',\n               '-DOKC_NS=3', '-DOKC_NSF=3', '-DOKC_PRMTSC=1', '-DOKC_SC6W=' + str(args.sc6), *args.define"
 assert driver.count(old)==1;driver=driver.replace(old,'*args.define')
 ns={'__name__':'_comment_family_g_compile'};exec(compile(driver,'<established-device-driver-with-actual-flags>','exec'),ns)
 environments={k:bindings((frozen/('G-'+k+'.py')).read_text(encoding='utf8')) for k in ('original','comment')}
 flags=dict(main=['-DOKC_NW=16','-DOKC_FAST1=0','-DOKC_EPI4=0','-DOKC_WGS=1','-DOKC_SC6W=1'],b4=['-DOKC_NW=16','-DOKC_FAST1=0','-DOKC_EPI4=0','-DOKC_WGS=1','-DOKC_B4FIN=1','-DOKC_SC6W=1'])
 for name,key in [('tiny','MOE_TINY_DEFINES'),('tiny2','MOE_TINY2_DEFINES'),('helper','MOE_MID_DEFINES')]:
  # Last definitions append only source selector and FINPF to these literal bases.
  f=list(environments['original'][key]);f=[x for x in f if not x.startswith('@')]
  if name=='tiny' and '-DOKC_FINPF=1' not in f:f.append('-DOKC_FINPF=1')
  flags[name]=f
 records=[];out.mkdir(parents=True,exist_ok=True)
 for name,f in flags.items():
  pair=[]
  for arm,env in environments.items():
   cuda=env['MOE_CUDA_SRC'] if name in ('main','b4') else env['_MA_ALT_SRCS']['moe_helper0']
   folder=out/(name+'-'+arm);folder.mkdir(parents=True,exist_ok=True);source=folder/'input.cu';source.write_bytes(cuda.encode())
   actual=['-DOKC_NS=3','-DOKC_NSF=3','-DOKC_FNST=4','-DOKC_PRMTSC=1','-DOKC_DQ7=1']+f
   a=types.SimpleNamespace(source=source,out=folder,cache=Path('/root/.cache/okc-cuda128'),sc6=1,define=actual,forward_declare_wgs=True)
   ns['compile_source'](a)
   disasm=subprocess.run([str(a.cache/'cuda_nvdisasm-linux-x86_64-12.8.90-archive/bin/nvdisasm'),str(folder/'device.cubin')],capture_output=True,text=True,check=True)
   (folder/'device.sass').write_text(disasm.stdout)
   pair.append((folder,json.loads((folder/'manifest.json').read_text()),sections(folder/'device.sass')))
  p,q=pair;assert p[1]['options']==q[1]['options'];assert p[1]['resources']==q[1]['resources'],(name,'resource drift')
  entries=[]
  for symbol in p[1]['lowered']:
   assert p[2][symbol]==q[2][symbol],(name,symbol,'SASS drift')
   entries.append(dict(symbol=symbol,instructions=len(p[2][symbol]),opcode_operand_sha256=hashlib.sha256('\n'.join(p[2][symbol]).encode()).hexdigest(),resources=p[1]['resources'][symbol]))
  records.append(dict(build=name,flags=p[1]['options'],entries=entries,original_cuda_sha256=p[1]['source_sha256'],comment_cuda_sha256=q[1]['source_sha256']))
  print('PAIR_IDENTICAL',name,flush=True)
 (out/'comparison.json').write_text(json.dumps(dict(records=records,GPU=False,scope='Five selected native build keys, two bool entry instantiations per key; device-only adapters omit host linking/launch'),indent=2)+'\n')

def reconcile(out,ledger):
 data=json.loads((out/'comparison.json').read_text());actual=json.loads(ledger.read_text())
 calls=next(r['captures'] for r in actual['records'] if r['family']=='G')
 def defines(flags):
  result={}
  for f in flags:
   if f.startswith('-D'):
    name,_,value=f[2:].partition('=');result[name]=value or '1'
  return result
 rows=[]
 for record in data['records']:
  matches=[c for c in calls if defines(c['flags'])==defines(record['flags'])]
  assert len(matches)==1,(record['build'],'actual flag match',len(matches))
  c=matches[0]
  assert c['raw_original_source_sha256']['cuda_sources']==[record['original_cuda_sha256']],(record['build'],'original source link')
  assert c['raw_packed_source_sha256']['cuda_sources']==[record['comment_cuda_sha256']],(record['build'],'packed source link')
  original=out/(record['build']+'-original');packed=out/(record['build']+'-comment')
  p=json.loads((original/'manifest.json').read_text());q=json.loads((packed/'manifest.json').read_text())
  assert p['ptx_launch_bounds']==q['ptx_launch_bounds'],'launch bounds drift'
  op=sections(original/'device.sass');oq=sections(packed/'device.sass');assert op==oq,(record['build'],'emitted helper text drift')
  used=lambda path:re.findall(r'^ptxas info\s+: Used .+$',path.read_text(),re.M)
  assert used(original/'ptxas.log')==used(packed/'ptxas.log'),'PTXAS resource line drift'
  rows.append(dict(build=record['build'],actual_original_module=c['original_name'],actual_packed_module=c['packed_name'],defines=defines(c['flags']),source_hashes_match=True,all_emitted_text_functions=len(op),all_emitted_text_sha256=hashlib.sha256(json.dumps(op,sort_keys=True).encode()).hexdigest(),launch_bounds_and_all_ptxas_used_resources_match=True))
 result=dict(ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),records=rows,scope='Exact full CUDA source byte hashes and effective defines match actual CPU-captured production build requests; no H200 claim')
 (out/'actual-builder-link.json').write_text(json.dumps(result,indent=2)+'\n');print('ACTUAL_BUILDER_SOURCE_FLAGS_LINKED',len(rows))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('frozen',type=Path);p.add_argument('out',type=Path);p.add_argument('--reconcile',type=Path);a=p.parse_args()
 if a.reconcile:reconcile(a.out,a.reconcile)
 else:run(a.root,a.frozen,a.out)
