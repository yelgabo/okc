"""607 layer applicability and conditional 46/16 sensitivity for owned SX prototype.
No port, source change, GPU, timing forecast or standalone-gain transfer.
"""
import argparse,ast,hashlib,json,math,re
from pathlib import Path
G_PIN='ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
JOB='607eb52bdaf24160'
HOST={'xor_build','xor_pack','_xor_pack_bits','xor_runner','bf16_bits','tile_values','window_base','code_bits','comp_units','stage_stream','balance_perm','slices_balanced','escape_table','pack_layer'}
def constants(s):
 t=ast.parse(s);out={}
 for n in t.body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
   try:out[n.targets[0].id]=ast.literal_eval(n.value)
   except (ValueError,TypeError):pass
 return t,out

def component(line,T):
 if 'sk2_kernel_gle' in line:return 'pool_router_fused'
 m=re.search(r'sk2_kernel(?:_narrow_\w+)?<([\d, ]+)',line)
 if not m:return None
 p=[int(v.strip()) for v in m[1].rstrip(', ').split(',')]
 if len(p)<7:return None
 mt,tokr,nt,nwr,wsk,upw,ns=p[:7]
 if nt==3:return 'qkv'
 if nt==1 and wsk==8:return 'un'
 if wsk==8:return 'pool_router_fused'
 if T in (1,8):return 'o'
 if T==16:return 'o' if ns==5 else 'un'
 if T==32:return 'o' if nt==2 else 'un'
 return None

def factor(rows,savings):
 out=[]
 for T in (1,16,32,8,128,256):
  pair=[r for r in rows if r['T']==T];assert len(pair)==2
  old=sum(r['weight']*r['median_us'] for r in pair)
  new=sum(r['weight']*(r['median_us']-savings(r)) for r in pair)
  assert new>0
  out.append(old/new)
 return math.prod(out)**(1/6)

def required(rows,save):
 lo,hi=0.,1.
 while factor(rows,lambda r:save(r)*hi)<1.01:hi*=2;assert hi<1000
 for _ in range(70):
  mid=(lo+hi)/2
  if factor(rows,lambda r:save(r)*mid)<1.01:lo=mid
  else:hi=mid
 return hi

def run(root,candidate):
 gp=root/'muse_decode_layer/cand/g1b4_stage_release.py';raw=gp.read_bytes();s=raw.decode();assert hashlib.sha256(raw).hexdigest()==G_PIN
 tree,g=constants(s);assert g['GLE_T']=={} and g['PDL'] is False and g['GLE_FRT'] is True
 receipt=root/f'muse_decode_layer/cand/profiles/g1b4_stage_release.{JOB}.json';d=json.loads(receipt.read_text(encoding='utf8'))
 assert d['job_id']==JOB and len(d['benchmarks'])==12
 profiles={(p['case']['load'],p['case']['layer']):p['report'] for p in d['profiles']}
 rows=[]
 for b in d['benchmarks']:
  c=b['case'];T=c['batch']*c.get('q_len',1);p=profiles[c['load'],c['layer']];comps={};lines=[]
  for line in p.splitlines():
   k=component(line,T);m=re.search(r'([\d.]+)\s*µs',line)
   if k and m:comps[k]=float(m[1]);lines.append(line)
  if T<=32:assert set(comps)=={'qkv','o','un','pool_router_fused'},(T,comps)
  rows.append({'load':c['load'],'T':T,'layer':c['layer'],'weight':c['weight'],'median_us':b['median_ns']/1000,'baseline_us':b['baseline_ns']/1000,'profile_components_us':comps,'profile_identities':lines})
 policy=[]
 cfg={c[0]:c for c in g['bfp_SK2_CONFIGS']}
 for T in (1,8,16,32):
  tk=8 if T<=8 else T
  for name,layer in (('qkv','qkv_gate'),('o','o'),('un','unpooler')):
   cname=g['bfp_SK2_PICK'][layer,tk];c=cfg[cname]
   target=('qkv8F' if T<=8 else 'qkv16F' if T==16 else 'qkv32') if name=='qkv' else ('o16F' if T<=16 else 'o32F') if name=='o' else ('slice8F' if T<=8 else 'unp16F' if T==16 else 'unp32F')
   policy.append({'T':T,'projection':name,'G_SK2_config':cname,'G_config_fields':c,'G_special':'qkv narrow2<=T<=8, ring/global-X escape modification' if name=='qkv' and T==8 else None,'SX_candidate':target,'SX_nearby_non_FULL_required':T==1 or (name=='o' and T==8),'input_output':'BF16 input→BF16 output','N':2304 if name=='qkv' else 1024 if name=='un' and T<=8 else 8192,'K':8192 if name=='qkv' else 1024 if name=='o' else 4096,'scope':'projection-boundary replacement requiring new SX packs/module; not source-identical G pipeline'})
 ct,cs=constants(candidate.read_text(encoding='utf8'));cuda=cs['_XOR_CUDA'];cpp=cs['_XOR_CPP'];assert 'COMP && !HL && CL==1 && !FUSE && !ESCG' in cuda
 host={n.name:len(ast.get_source_segment(candidate.read_text(encoding='utf8'),n).encode()) for n in ct.body if isinstance(n,ast.FunctionDef) and n.name in HOST}
 assert set(host)==HOST,(set(host)^HOST)
 meta=len(repr(cs['_XOR_CONFIG']).encode());native_visible=len(('r\'\'\'\n'+cuda+'\n\'\'\'').encode());hostbytes=sum(host.values());budget=720525+native_visible+len(cpp.encode())+hostbytes+meta+4096+2048
 scenarios=[]
 for pct in (1,2,5,10):
  scenarios.append({'uniform_percent_of_qkv_o_un_profile_time_removed':pct,'conditional_full_six_factor':factor(rows,lambda r:sum(r['profile_components_us'].get(k,0) for k in ('qkv','o','un'))*pct/100)})
 return {'G_source_sha256':G_PIN,'G_source_bytes':len(raw),'job':JOB,'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'candidate_artifact_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'candidate_cuda_sha256':hashlib.sha256(cuda.encode()).hexdigest(),'PDL':False,'GLE_T':{},'cells':rows,'dense_policy':policy,'not_immediate':['T1/8 GLE_FRT: router reads FP32 hr, pool BF16; fused lifecycle must remain','T16/32 bfp_fused_pooler_router: FUSE true pooler plus HL router continuation; SX forbids both','T128/256 Lt projections remain unchanged'],'one_percent_six_load_requirements':{'same_whole_cell_us_saved_at_T1_8_16_32':required(rows,lambda r:float(r['T']<=32)),'same_us_per_qkv_o_un_projection':required(rows,lambda r:3.*float(r['T']<=32)),'uniform_fraction_of_qkv_o_un_profile_time':required(rows,lambda r:sum(r['profile_components_us'].get(k,0) for k in ('qkv','o','un')))},'scenarios':scenarios,'byte_plan':{'factored_G_raw':720525,'SX_full_native_visible_literal':native_visible,'SX_CPP_literal_value':len(cpp.encode()),'unchanged_host_function_source_total':hostbytes,'host_function_source_bytes':host,'all10_config_repr':meta,'adapter_import_glue_allowance':4096,'additional_notice_allowance':2048,'conservative_source_plan_sum':budget,'not_a_built_layer_artifact':True},'scope':'Actual607medians; profile-based hypothetical removal only, not paired timings, wait bounds or standalone transfer; no port/GPU'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.root,a.candidate);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf8');print(json.dumps({k:r[k] for k in ('one_percent_six_load_requirements','scenarios','byte_plan')},indent=2))
