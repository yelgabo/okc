"""Private measurement-only BF16-output-matched VP2 comparator; no donor adoption."""
import argparse,ast,hashlib,importlib.util,json
from pathlib import Path

OWN='0627edb55a4f28e8943c9e605246757287cde74f54ddc4306e0c8cca81457ea4'
PUBLIC='496d460e0bb1e7489684de2742f1ff22f3959d9d5fe30674199e74b3d002d651'
SEEDS=(824731,824737,824743)

def once(s,a,b):
 assert s.count(a)==1,(a[:80],s.count(a));return s.replace(a,b)

RESOURCE=r'''
std::vector<int64_t> vp2_resources(){
 using namespace vfgns::b1vp2;
 cudaFuncAttributes a{};int clusters=0,blocks=0;
 const int smem=sizeof(VP2Smem)+1024;
 TORCH_CHECK(cudaFuncSetAttribute(vp2_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem)==cudaSuccess,"VP2 shared opt-in");
 TORCH_CHECK(cudaFuncSetAttribute(vp2_kernel,cudaFuncAttributeNonPortableClusterSizeAllowed,1)==cudaSuccess,"VP2 cluster opt-in");
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(32,1,1);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=smem;
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=16;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;
 cfg.attrs=&attr;cfg.numAttrs=1;
 TORCH_CHECK(cudaFuncGetAttributes(&a,vp2_kernel)==cudaSuccess,"VP2 attrs");
 TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,vp2_kernel,256,smem)==cudaSuccess,"VP2 blocks");
 TORCH_CHECK(cudaOccupancyMaxActiveClusters(&clusters,vp2_kernel,&cfg)==cudaSuccess && clusters>=2,"VP2 two active clusters");
 TORCH_CHECK(a.localSizeBytes==0,"VP2 local-memory gate");
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,smem,256,blocks,clusters};
}
'''
OWN_RESOURCE=r'''
std::vector<int64_t> vp_owned_resources(){
 std::vector<int64_t> r;
#define VR(L,M) {Instance<L,M>::prepare();cudaFuncAttributes a{};TORCH_CHECK(cudaFuncGetAttributes(&a,attn_one_kernel<L,M>)==cudaSuccess,"VP owned attrs");for(int64_t v:{int64_t(L),int64_t(M),int64_t(a.numRegs),int64_t(a.localSizeBytes),int64_t(Instance<L,M>::BYTES),int64_t(Instance<L,M>::NT),int64_t(Instance<L,M>::per_sm)})r.push_back(v);}
 ONE_INSTANCES(VR)
#undef VR
 return r;
}
'''

def literal(tree,name):
 return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')==name for t in n.targets))

def make(root,output):
 src=(root/'muse_attention_decode/submission.py').read_text(encoding='utf8')
 assert hashlib.sha256(src.encode()).hexdigest()==OWN
 tree=ast.parse(src);cu=literal(tree,'_ONE_SRC');cpp=literal(tree,'_ONE_CPP')
 assert cu.count('acc[e] *= inv;')==2
 cu=cu.replace('acc[e] *= inv;','acc[e]=__bfloat162float(__float2bfloat16_rn(acc[e]*inv));')+OWN_RESOURCE
 # Every device function stays in the literal. Only four import-time extension builds are
 # deferred/replaced; private non-TP8 control explicitly uses existing SGLang fallback.
 replacements=[]
 for n in tree.body:
  if isinstance(n,ast.Try):
   assert isinstance(n.body[0],ast.Assign)
   name=n.body[0].targets[0].id;assert name in ('_ONE','_PREP','_ATTN','_WS')
   replacements.append((n.lineno,n.end_lineno,name+' = None\n'))
 lines=src.splitlines(keepends=True)
 for a,b,text in reversed(replacements):lines[a-1:b]=[text]
 owned=''.join(lines);t=ast.parse(owned)
 replacements=[]
 for n in t.body:
  if isinstance(n,ast.Assign):
   name=getattr(n.targets[0],'id','')
   if name in ('_ONE_SRC','_ONE_CPP'):
    replacements.append((n.lineno,n.end_lineno,name+' = '+repr(cu if name=='_ONE_SRC' else cpp+'\nstd::vector<int64_t> vp_owned_resources();\n')+'\n'))
 lines=owned.splitlines(keepends=True)
 for a,b,text in reversed(replacements):lines[a-1:b]=[text]
 owned=''.join(lines);ast.parse(owned)
 donor=root/'.build/today/kel/kernels/muse_attention_decode.py';raw=donor.read_bytes()
 assert hashlib.sha256(raw).hexdigest()==PUBLIC
 dt=ast.parse(raw.decode());vcu=literal(dt,'ZH_VFG_CUDA');vcpp=literal(dt,'ZH_VFG_CPP')
 vcu=once(vcu,'      b0 *= inv; b1 *= inv;\n      float ss = b0 * b0 + b1 * b1;',
  '      b0=__bfloat162float(__float2bfloat16_rn(b0*inv)); b1=__bfloat162float(__float2bfloat16_rn(b1*inv));\n      float ss = b0 * b0 + b1 * b1;')
 for text in ('cudaFuncSetAttribute(vp2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);','cudaFuncSetAttribute(vp2_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);'):
  vcu=once(vcu,text,'TORCH_CHECK('+text[:-1]+'==cudaSuccess,"VP2 launch attribute");')
 vcu+=RESOURCE;vcpp+='\nstd::vector<int64_t> vp2_resources();\n'
 spec=importlib.util.spec_from_file_location('_vp_ref_extract',root/'tools/attn_make_bench_one.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 runner=mod.RUNNER.format(variants=[],coop_probe=True);rt=ast.parse(runner)
 keep={'_OB_HEADS','_OB_WINDOW','_OB_THETA'};funcs={'_ob_rms','_ob_rope','_ob_inputs','_ob_reference','_ob_rel'}
 refs='\n'.join(ast.get_source_segment(runner,n) for n in rt.body if
   isinstance(n,ast.FunctionDef) and n.name in funcs or isinstance(n,ast.Assign) and any(getattr(t,'id','') in keep for t in n.targets))
 # Separate baseline-output oracle. The original FP32 judge oracle above is untouched.
 bfr=once(next(ast.get_source_segment(runner,n) for n in rt.body if isinstance(n,ast.FunctionDef) and n.name=='_ob_reference'),
  'def _ob_reference(', 'def _vp_bf_reference(')
 bfr=once(bfr,'    out = _ob_rms(out) * torch.sigmoid(gate.float())','    out = _ob_rms(out.to(torch.bfloat16)) * torch.sigmoid(gate.float())')
 refs+='\n'+bfr+'\n'
 notice=(root/'.build/today/kel/NOTICE').read_text(encoding='utf8')
 license=(root/'.build/today/kel/LICENSE').read_text(encoding='utf8')
 manifest={'owned_lf':OWN,'public_raw':PUBLIC,'public_commit':'8e30e1be994db4310794b8d04e90d18d895deb84',
  'owned_cuda':hashlib.sha256(cu.encode()).hexdigest(),'vp2_cuda':hashlib.sha256(vcu.encode()).hexdigest(),
  'notice_sha256':hashlib.sha256(notice.encode()).hexdigest(),'license_sha256':hashlib.sha256(license.encode()).hexdigest(),
  'seeds':SEEDS,'policy':'PRIVATE judged comparator VP2 only B1/q8/sliding/page1/unitKV; all other scored cells actual BF16 owned ONE; nonTP8 existing SGLang fallback',
  'measurement_only':True,'sch':0,
  'decision_rule':'THIS official job raw-minus-adjusted offsets applied commonly to private raw arms; contemporaneous >=1% and >2x null pooled and every seed, no regression/no nonpositive times; raw all4 crosses positive every seed and no regression. Old-offset/raw keep flags descriptive only.',
  'build_ids':['vp2bf_owned824731_0','vp2bf_owned824731_1','vp2bf_foreign824731_2','vp2bf_foreign824731_3'],
  'error_order':['FP32judge_global','FP32judge_row','BF16output_global','BF16output_row','newK','newV']}
 calibration=root/'muse_attention_decode/profiles/wait-trace.3f44f891d0ef47db.json'
 records=json.loads(calibration.read_text(encoding='utf8'))['benchmarks'];offsets={}
 for r in records:
  c=r['case'];length=8 if c['load'].startswith('verify') else 1
  offsets[(c['batch'],length,int(c['layer']=='sliding'))]=(r['median_raw_ns']-r['median_ns'])/1000
 assert len(offsets)==12
 manifest['calibration_receipt_sha256']=hashlib.sha256(calibration.read_bytes()).hexdigest()
 header='# okc: problem=muse_attention_decode\n# PRIVATE MEASUREMENT ONLY: no public submission, no donor adoption.\n'
 out=header+'\n_VP_LICENSE='+repr(license)+'\n_VP_NOTICE='+repr(notice)+'\n'
 out+='_VP_OWNED='+repr(owned)+'\n_VP_CUDA='+repr(vcu)+'\n_VP_CPP='+repr(vcpp)+'\n_VP_MANIFEST='+repr(manifest)+'\n_VP_OFFSETS='+repr(offsets)+'\n'
 out+='import torch,time,math,statistics,concurrent.futures\nRMS_EPS=1e-5\nFP8=torch.float8_e4m3fn\n'+refs
 out+=Path(__file__).with_name('moe_attn_vp2_body.py').read_text(encoding='utf8')
 ast.parse(out);assert len(out.encode())<1_000_000
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(out,encoding='utf8',newline='\n')
 output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
 output.with_suffix('.vp2.cu').write_text(vcu,encoding='utf8',newline='\n')
 output.with_suffix('.owned.cu').write_text(cu,encoding='utf8',newline='\n')
 print('VP2_PRIVATE',len(out.encode()),hashlib.sha256(out.encode()).hexdigest())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();make(a.root,a.output)
