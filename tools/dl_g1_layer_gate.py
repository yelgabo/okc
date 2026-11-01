"""Generate next-reset whole-layer G1b4/stage-retire gate. ZERO dispatch.

Example AFTER the normal G1b4 conformance receipt passes:
  python tools/dl_g1_layer_gate.py .build/today/g1b4_native_experts.py \
    .build/today/stage-layer-decode.py --root . --part decode \
    --g1b4-receipt path/to/g1b4-terminal.json
Repeat with --part verify, then use dl_g1_layer_merge.py on BOTH terminal
candidate-judged receipts. --part all is a fail-closed300s attempt if practical.
Without --g1b4-receipt the artifact is source/CPU-reviewable but cannot measure.

One preparation; fourprivate alias-preserving owners. Inventoried read-only
weights/packed tables shared, all mutable storage and nested Lt workspace cloned.
Unknown owners, overlap, >8GiB mutablecopies or insufficient free memory fail.
All36heldout rows+13external official candidate tests are required for one score
decision. Mixedrouting seeds choose busiest/second/least rank. Each fresh bank has
8replays with final math check; this is NOT every-intermediate-replay validation.
Actual H200 compile time, graph resource equality, owner completeness and runtime
have NOT been measured for this harness. No fallback, silent partial coverage or
standalone-to-layer extrapolation can qualify it. Pure candidate remains separate.
"""
import argparse,ast,hashlib,importlib.util,sys,json
from pathlib import Path
PIN='1607293eee3403109df5d37b6e2d1ee34b528c3578afa6c2c8668440afde1658'
NAMES=('_ab_inputs','_ab_rms','_ab_rope','_ab_coeffs','_ab_attention','_ab_ref_front','_ab_deq','_ab_ref_back','_ab_local','_ab_check_routing','_ab_compare','_ab_validate')
PRELUDE='''
import time
FP8=torch.float8_e4m3fn
RMS_EPS=1e-5
POST_NORM_EPS=1e-8
ROUTER_MULTIPLIER=0.1767766952966369
SLICED_UNPOOL_MAX_TOKENS=8
UNPOOL_SLICE=1024
_E2M1=[0.,.5,1.,1.5,2.,3.,4.,6.]
_AB_DEQ={}
_REF_STREAM_SEED,_REF_STREAM_MEAN=5354,0.7
_REF_STREAM_SHARED,_REF_STREAM_RANK,_REF_DRAFT_SHARE=.9,4,.6
'''

def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def conformance(path):
    if path is None:return None
    obj=json.loads(path.read_text(encoding='utf8'));r=obj['result']
    assert obj['board']=='muse_decode_layer' and obj['status']=='done' and r['passed'] and r['checks']['ok']
    tests=[x for x in r['records'] if x.get('event')=='test']
    assert len(tests)==13 and all(x['status']=='pass' for x in tests),'13 baseline tests required'
    assert 'EMPTY_CONFORMANCE_ALL' in r['log_tail'] and 'G1b4 backend' in r['log_tail'],'G1b4 native conformance transcript required'
    return {'job_id':obj['id'],'candidate_pin':PIN,'status':'done_pass13_native_empty','evaluator_code':r['evaluator']['code_sha256']}

def build(source,root,part,proof=None):
    assert hashlib.sha256(source.encode()).hexdigest()==PIN
    sys.path.insert(0,str(root/'tools'))
    stage=load(root/'tools/dl_g1_stage_retire.py','_wl_stage')
    resource=load(root/'tools/moe_make_sc6_compare.py','_wl_resource').RESOURCE.replace('sc6_resources','wl_resources')
    tree=ast.parse(source);n=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='MOE_CUDA_SRC')
    device=stage.patch_device(ast.literal_eval(n.value),switch=True)+resource
    lines=source.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['MOE_CUDA_SRC = '+repr(device)+'\n'];s=''.join(lines)
    s=once(s,'    ext = state["ma_ext"].get(key)','    if state.get("_wl_stage_on",False) and "-DOKC_WGS=1" in defines:\n        defines=tuple(defines)+("-DOKC_STAGE_RETIRE=1",)\n    ext = state["ma_ext"].get(key)')
    s=once(s,'std::vector<int64_t> okc_attrs();','std::vector<int64_t> okc_attrs();\nstd::vector<int64_t> wl_resources();')
    s=once(s,'functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []),','functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []) + (["wl_resources"] if "wl_resources" in src else []),')
    body=(root/'tools/dl_make_bench_pub.py').read_text(encoding='utf8');body=next(ast.literal_eval(n.value) for n in ast.parse(body).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='BODY')
    funcs={n.name:ast.get_source_segment(body,n) for n in ast.parse(body).body if isinstance(n,ast.FunctionDef)}
    assert set(NAMES)<=funcs.keys()
    # The reference helpers reproduce FP32 judge math; do not repair to SGLang BF16 here.
    helpers='\n\n'.join(funcs[n] for n in NAMES)
    local=Path(__file__).parent
    loads={'all':(1,8,16,32,128,256),'decode':(1,16,32),'verify':(8,128,256)}[part]
    core=(local/'dl_g1_layer_owners.py').read_text(encoding='utf8')+(local/'dl_g1_layer_body.py').read_text(encoding='utf8')+helpers+PRELUDE
    identity='\n_WL_DEVICE_SHA='+repr(hashlib.sha256(device.encode()).hexdigest())+'\n_WL_CORE_SHA='+repr(hashlib.sha256(core.encode()).hexdigest())+'\n'
    out=s+identity+'\n_WL_CONFORMANCE='+repr(proof)+'\n'+PRELUDE+'\n'+helpers+'\n'+(local/'dl_g1_layer_owners.py').read_text(encoding='utf8')+'\n_WL_PART='+repr(part)+'\n_WL_LOADS='+repr(loads)+'\n_WL_PIN='+repr(PIN)+'\n'+(local/'dl_g1_layer_body.py').read_text(encoding='utf8')
    ast.parse(out);assert len(out.encode())<1_000_000
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--root',type=Path,required=True);p.add_argument('--g1b4-receipt',type=Path);p.add_argument('--part',choices=('all','decode','verify'),default='all');a=p.parse_args()
    s=build(a.source.read_text(encoding='utf8'),a.root,a.part,conformance(a.g1b4_receipt));a.output.write_bytes(s.encode());print('WL_GATE',a.part,len(s.encode()),hashlib.sha256(s.encode()).hexdigest())
