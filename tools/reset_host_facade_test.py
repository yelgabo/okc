"""Whole host prepare CPU facade. Native arithmetic, CUDA capture/events/profiler are explicit stand-ins."""
import ast,base64,contextlib,hashlib,io,json,os,sys,time,types,unittest,zlib
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[1]))
OWNERS=Path(__file__).with_name('dl_g1_layer_owners.py')
class Module(types.ModuleType):
 def wl_resources(self):return [96,0,231168,128,1,16]*2
 def okc_run(self,*a):a[5].zero_()
class _RouterExt:
 def __init__(self,m):self._module=m
 def __getattr__(self,k):return getattr(self._module,k)
class _GraftLt:pass
class _GraftPlan(dict):pass
# Use the actual class definitions; skip only native handle constructors.
_source=(ROOT/'muse_decode_layer/cand/g1b4_control_release.py').read_text(encoding='utf8')
_tree=ast.parse(_source)
for _node in _tree.body:
 if isinstance(_node,ast.ClassDef) and _node.name in ('_RouterExt','_GraftLt','_GraftPlan'):
  exec(ast.get_source_segment(_source,_node),globals())
class Stream:
 def wait_stream(self,x):pass
class Event:
 def __init__(self,**kw):pass
 def record(self):pass
 def synchronize(self):pass
 def elapsed_time(self,z):return .1
class Cuda:
 def mem_get_info(self):return 100<<30,140<<30
 def synchronize(self):pass
 def Event(self,**kw):return Event()
class Torch:
 Tensor=torch.Tensor;cuda=Cuda()
 def __getattr__(self,k):return getattr(torch,k)
 def empty(self,*a,**kw):
  if a==(256<<20,):return torch.empty(1,dtype=torch.uint8)
  return torch.empty(*a,**kw)
def state():
 t=lambda:torch.arange(8,dtype=torch.float32)
 lt=_GraftLt.__new__(_GraftLt)
 for k in ('lib','handle','desc','alpha','beta'):setattr(lt,k,object())
 lt.workspace=t();plan=_GraftPlan(lt=lt,weight=t(),layouts=(object(),object(),object()),algo=object(),n=4096,fp32=False)
 shared_set={('pooler',8),(True,None,3.25,'bfp')}
 s={'ma_w':(t(),t()),'qkv':t(),'ma_scratch':[t(),t(),t()], 'ma_ext':{},'attention_scratch':{'rows':{},'cluster':{},'retained':[]},'nsm':132,
 '_fork_native_eligible':True,'ma_ok':True,'ma_sc6':True,'bank':None,'router_fp32_ext':Module('router'),'graft_lt':{('qkv',32):plan},'graft':{},
 'compose_active':set(),'compose_selection':{},'compose_base_mm':{},'compose_base_graft_lt':{},'compose_graft_enabled':True,'compose_boundary_warps':8,
 'mm_pick':{},'moe_pick':{},'gluev2':{},'marlin_cfg':{},'pbmax':0,'fa3':None,'att_ok':True,'graft_lt_status':{},'_fork_moe_status':'NATIVE',
 'bfp':{k:{} for k in ('raw','cnt','comp','mid_cnt','streams','tabs','base4','sk2','fused_part','fused_cnt','mid_ws_shared','ws_shared','choice')}}
 s['bfp'].update(disabled=shared_set,dummy=t(),sm90=True,old_paths=False,ws_shared_diag=False,router_hi_t=t(),router_lo_t=t(),router_cat=t())
 s['bfp']['choice'][('pooler',8)]=('sk2',1,2);s['set_alias']=shared_set
 for k in ('raw','streams','tabs','base4') :s['bfp'][k][('pooler',1)]=t()
 for k in ('cnt','mid_cnt','fused_cnt','mid_ws_shared','ws_shared'):s['bfp'][k][('pooler',1)]=t()
 s['bfp']['comp']['pooler']=True;s['bfp']['sk2'][('pooler',1)]=(1,t(),t(),2,t(),4096)
 s['bfp']['fused_part'][('pooler',1)]=(t(),t())
 for k in ('o','pr','unpool','unpool_slice','gate1','gate2','g_preffn','g_preexp','g_postexp','g_postffn','g_next','bias','gs13','gs2','invf0','invf','router_fp32'):s[k]=t()
 s['ma_sfx']=('-DOKC_PRMTSC=1',);s['_fork_fallback_kind']='NONE'
 return s
def inputs(b,l,c,seed):
 g=torch.Generator().manual_seed(seed);n=b*l
 return dict(config=dict(tokens=n,batch=b,q_len=l,context=c),stream=torch.randn(n,4,generator=g),attn_input=torch.randn(n,4,generator=g),k_cache=torch.zeros(n,4),v_cache=torch.zeros(n,4),local_start=torch.zeros(1,dtype=torch.int32))
def front(d,w,*args):
 n=d['config']['tokens'];v=torch.arange(256,dtype=torch.float32).expand(n,256).clone();v[:,0:8]+=1000
 return dict(biased=v,local_start=int(d['local_start'][0]))
def validate(base,out,d,w,*refs):
 okay=all(torch.equal(o,e) for o,e in zip(out,(base['stream']+base['attn_input'],base['attn_input']*2,refs[0]['biased'].topk(8,dim=-1).indices.to(torch.int32))))
 return okay,'CPU arithmetic stand-in'
class Graph:
 def __init__(self,fn):self.fn=fn
 def replay(self):self.fn()
class Profile:
 def __init__(self,cuda):self.cuda=cuda;self.profiler=types.SimpleNamespace(kineto_results=types.SimpleNamespace(events=self.events))
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def events(self):
  from torch.autograd import DeviceType
  names=['qkv','attn','o','glue_cluster_boundary']+(['gle_frt'] if self.cuda.mode=='fused' else ['pooler','router_fp32_kernel'])+['_sl_tile_route','okc::moe_fused','unpool']
  return [types.SimpleNamespace(start_ns=lambda i=i:i*2000,end_ns=lambda i=i:i*2000+1000,device_type=lambda:DeviceType.CUDA,device_resource_id=lambda:1,name=lambda n=n:n) for i,n in enumerate(names)]
def fixture(kind,old=False):
 ns=dict(torch=Torch(),time=time,prepare=lambda w:state(),kernel_fn=lambda *a:None,_fork_native_moe=lambda *a:None,_decode_attention7=lambda *a:None,_ma_ext=lambda *a:None,_pool_router_fp32=lambda *a:None,_GUARDS=[],_pdl=lambda t:False,PDL=False,GLE_FRT=True,FP8=torch.float32,TOP_K=8,LOCAL_EXPERTS=32,SCRATCH_SENTINEL=0,_AB_DEQ={},_check_guards=lambda:None,_ab_inputs=inputs,_ab_ref_front=front,_ab_validate=validate)
 owner=OWNERS.read_text()
 if old:
  a=owner.index('        if type(x) is set:');b=owner.index("        if type(x).__name__=='_GraftLt':",a);owner=owner[:a]+owner[b:]
 exec(owner,ns)
 exec((ROOT/'tools'/('dl_g1_layer_body.py' if kind=='stage' else 'dl_short_path_body.py')).read_text(),ns)
 ns.update(_WL_PIN='test',_WL_CONFORMANCE={'candidate_pin':'test'},_WL_PART='all',_WL_LOADS=(1,8,16,32,128,256),_WL_DEVICE_SHA='stand-in',_WL_CORE_SHA='stand-in')
 modules={}
 def native(n,s):
  key='b4' if 16<n<=32 else 'main' if n>32 else 'small';on=s.get('_wl_stage_on',False) and key!='small';ident=(key,on)
  if ident not in modules:modules[ident]=_RouterExt(Module(str(ident)))
  e=modules[ident];s['ma_ext'][key]=e
  flags=('-DOKC_STAGE_RETIRE=1',) if on else ()
  if key=='b4':flags+=('-DOKC_B4FIN=1',)
  ns.setdefault('_WL_EXT_FLAGS',{})[id(e._module)]=flags
  return e
 ns['_fork_native_moe']=native
 def capture(d,s):
  n=d['config']['tokens'];native(n,s);s['_wl_hits']=[(n,'b4' if n==32 else 'main' if n>32 else 'small',s.get('_wl_stage_on',False))]
  ids=front(d,None)['biased'].topk(8,dim=-1).indices.to(torch.int32)
  out=(torch.empty_like(d['stream']),torch.empty_like(d['stream']),ids.clone())
  def replay():
   out[0].copy_(d['stream']+d['attn_input']);out[1].copy_(d['attn_input']*2);out[2].copy_(front(d,None)['biased'].topk(8,dim=-1).indices)
   d['k_cache'].copy_(d['attn_input']);d['v_cache'].copy_(d['stream']);ns['torch'].cuda.mode=s.get('_sp_mode','fused')
  g=Graph(replay);g.replay()
  if kind=='stage':return g,out
  lids=ids-int(d['local_start'][0]);lids=torch.where((lids>=0)&(lids<32),lids,-1)
  s['_sp_capture']=dict(ids=ids,lids=lids);s['_sp_native']=[native(n,s)._module]
  return g,out,(d['stream'].clone(),d['attn_input'].clone())
 ns['_wl_capture']=capture;ns['_sp_graph']=capture
 ns['summarize']=lambda events:dict(span_us=(max(v[1] for v in events)-min(v[0] for v in events))/1000,events=events)
 ns['encode_report']=lambda r:base64.b64encode(zlib.compress(json.dumps(r).encode(),9)).decode()
 return ns
class Tests(unittest.TestCase):
 def test_source_owner_inventory(self):
  # This is an assignment-site census, not a dump of an H200 prepared state.
  assignments={}
  for n in ast.walk(_tree):
   if isinstance(n,ast.Assign):
    for target in n.targets:
     if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id in ('state','st') and isinstance(target.slice,ast.Constant) and isinstance(target.slice.value,str):
      assignments.setdefault(target.slice.value,[]).append(dict(line=n.lineno,expression=ast.unparse(n.value)))
  for k in ('compose_active','bfp','ma_w','ma_scratch','ma_ext','graft_lt','router_fp32_ext','attention_scratch'):self.assertIn(k,assignments)
  self.assertTrue(any('set()' in r['expression'] for r in assignments['compose_active']))
  # Nested bfp disabled set is a separate literal assignment site.
  self.assertIn('disabled',assignments);self.assertTrue(any(r['expression']=='set()' for r in assignments['disabled']))
  self.assertIsInstance(state()['graft_lt'][('qkv',32)],_GraftPlan)
 def test_sets_alias_and_members(self):
  ns=fixture('stage');s=state();ro=ns['_wl_readonly'](s,{})
  a=ns['_wl_clone'](s,ro,False);b=ns['_wl_clone'](s,ro,True)
  self.assertIs(a['set_alias'],a['bfp']['disabled']);self.assertIsNot(a['set_alias'],s['set_alias']);self.assertIsNot(a['set_alias'],b['set_alias']);a['set_alias'].add('new');self.assertNotIn('new',s['set_alias']);ns['_wl_check_owners']([s,a,b],ro)
  for bad in (torch.tensor(1),object(),frozenset({1}),(torch.tensor(1),)):
   s['compose_active']={bad}
   with self.assertRaisesRegex(RuntimeError,'unreviewed policy set member'):ns['_wl_clone'](s,ro,False)
 def test_old_helper_fails_both_whole_prepares(self):
  for kind in ('stage','short'):
   ns=fixture(kind,True)
   with self.assertRaisesRegex(RuntimeError,'unreviewed state owner set'):ns['_wl_prepare' if kind=='stage' else '_sp_prepare']({'weight':torch.ones(1)})
 def test_stale_replay_and_budget_fail_closed(self):
  ns=fixture('stage');capture=ns['_wl_capture']
  def stale(d,s):
   graph,out=capture(d,s);graph.fn=lambda:None;return graph,out
  ns['_wl_capture']=stale
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(AssertionError,'WL_MATH'):ns['_wl_prepare']({'w':torch.ones(1)})
  for kind,guard in (('stage','WL_INCOMPLETE'),('short','SHORT_INCOMPLETE')):
   ns=fixture(kind);clock=iter((0,301));ns['time']=types.SimpleNamespace(monotonic=lambda:next(clock))
   with self.assertRaisesRegex(AssertionError,guard):ns['_wl_prepare' if kind=='stage' else '_sp_prepare']({'w':torch.ones(1)})
 def test_stage_whole_prepare(self):self.run_prepare('stage')
 def test_short_whole_prepare(self):self.run_prepare('short')
 def run_prepare(self,kind):
  ns=fixture(kind);f=io.StringIO();w={'weight':torch.ones(1)};tf32=torch.backends.cuda.matmul.allow_tf32;ns['_AB_DEQ']['sentinel']=True
  with patch('torch.profiler.profile',side_effect=lambda **kw:Profile(ns['torch'].cuda)),contextlib.redirect_stdout(f):result=ns['prepare'](w)
  self.assertEqual(torch.backends.cuda.matmul.allow_tf32,tf32);self.assertFalse(ns['_AB_DEQ'])
  line=[l for l in f.getvalue().splitlines() if l.startswith('WL_STAGE_REPORT ' if kind=='stage' else 'SHORT_PATH_TRACE ')][0]
  r=json.loads(zlib.decompress(base64.b64decode(line.split(' ',1)[1])));self.assertLess(len(line.split(' ',1)[1]),11000)
  if kind=='stage':self.assertEqual((len(r['rows']),r['math_checks'],r['fresh_final_checks'],r['empty_native_checks']),(36,144,144,12));self.assertTrue(result['_wl_stage_on'])
  else:self.assertEqual((len(r['rows']),r['checks']),(48,96));self.assertNotIn('_sp_active',result)
  self.assertEqual(result['bfp']['disabled'],{('pooler',8),(True,None,3.25,'bfp')})
  print(kind,'whole prepare CPU facade PASS',len(r['rows']),'rows')
if __name__=='__main__':unittest.main()
