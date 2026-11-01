"""Pinned CPU-only G0/607 dense route and contract inventory. No source variant or GPU."""
import argparse, ast, hashlib, json, re, subprocess, sys
from pathlib import Path
from prod_size_compile_g import bindings


def sha(b):return hashlib.sha256(b).hexdigest()


def audit(root):
    paths=[root/'.build/today/kel/kernels/muse_decode_layer.py',root/'muse_decode_layer/cand/g1b4_stage_release.py']
    raw=[p.read_bytes() for p in paths]
    pins=['1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a','ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab']
    assert list(map(sha,raw))==pins
    text=[b.decode().replace('\r\n','\n') for b in raw];env=[bindings(s) for s in text]
    for s,e in zip(text,env):
        for n in ast.parse(s).body:
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
                try:e[n.targets[0].id]=ast.literal_eval(n.value)
                except (ValueError,TypeError):pass
    equal_names=('bfp_SK2_CONFIGS','bfp_SK2_PICK','bfp_SK2_K','_GRAFT_LT_PICKS','_GRAFT_QKV256','_GRAFT_LT_T','bfp_COMPRESS')
    equals={k:env[0][k]==env[1][k] for k in equal_names};assert all(equals.values())
    funcs={}
    for name in ('dense','_graft_layouts','_graft_lt_run','_pool_router_fp32','bfp__sk2_geom','bfp__sk2_src'):
        nodes=[[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name][-1] for s in text]
        funcs[name]=dict(ast_equal=ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False),lines=[n.lineno for n in nodes])
    assert all(v['ast_equal'] for v in funcs.values())
    sys.path.insert(0,str(root/'tools'));from dl_proj_budget_audit import dense_rows
    files=['muse_decode_layer.3a5f879df0d44825.json','g1-stage-release.607eb52bdaf24160.json']
    data=[json.loads((root/'muse_decode_layer/profiles'/f).read_text(encoding='utf8')) for f in files]
    assert [d['job_id'] for d in data]==['3a5f879df0d44825','607eb52bdaf24160']
    configs={x[0]:list(x) for x in env[0]['bfp_SK2_CONFIGS']}
    rows=[]
    for i,b in enumerate(data[1]['benchmarks']):
        c=b['case'];t=c['batch']*c.get('q_len',1);assert all(c.get(k,1)==data[0]['benchmarks'][i]['case'].get(k,1) for k in ('load','layer','batch','q_len','weight'))
        profiles=[dense_rows(d['profiles'][i]['report']) for d in data]
        operators={}
        for op,layer in (('qkv','qkv_gate'),('o','o'),('pr','pooler'),('un','unpooler')):
            if t<=32:
                bucket=8 if t<=8 else t;cfg=env[1]['bfp_SK2_PICK'][(layer,bucket)]
                if op=='pr':
                    cfg='pooler';route='GLE_FRT pooler + full-FP32 router' if t<=8 else 'SK2F pooler + BF16 hi/lo router'
                else:route='narrow SK2' if op=='qkv' and t==8 else 'SK2'
                if op=='o' and t==1:route='G0 deferred GLE O+boundary;607 ordinary SK2 O then boundary'
                matches=[]
                for prof in profiles:
                    target=[]
                    for n,us,calls in prof:
                        args=n.split('<',1)[1].split(',')
                        nt,wsk=int(args[2]),int(args[4])
                        actual='qkv' if nt==3 else 'pr' if nt==4 and wsk==8 else 'un' if nt==1 else ('un' if t==16 and int(args[6])==5 or t==32 and nt==4 and int(args[3])==2 else 'o')
                        if actual==op:target.append(dict(kernel=n,aggregate_us=us,calls=calls))
                    assert len(target)==1,(t,op,target)
                    matches.append(target[0])
                operators[op]=dict(route=route,config_name=cfg,config=configs[cfg],profile=matches,
                    special='T8 QKV narrow override uses5 stages, not static qkv8 cap6' if op=='qkv' and t==8 else None,
                    numerical_change='none in projection MMA/weight codec; downstream mandatory boundary casts differ',
                    accumulation='FP32 MMA plus FP32 escape correction',output='FP32 pooler/router staging; BF16 pooler stage before norm, FP32 router logits' if op=='pr' else 'BF16')
            else:
                pick=env[1]['_GRAFT_QKV256'] if (op,t)==('qkv',256) else env[1]['_GRAFT_LT_PICKS'][(op,t)]
                old='rs128 '+('n256s5m1hzmEH' if t==256 else 'd2s5hzwP') if (op,t) in (('qkv',128),('qkv',256),('un',128)) else 'same saved Lt plan'
                operators[op]=dict(g0_route=old,g607_route='saved checked nonsplit cuBLASLt',plan=pick,
                    compute='CUBLAS_COMPUTE_32F (68)',operands='BF16',output='FP324608 concatenated pooler/router staging' if op=='pr' else 'BF16',
                    source_selection_verified=True,per_operator_profiler_attribution='grouped QKV+un rq inG0 or QKV+O Lt in607 atT128; do not split aggregate calls' if t==128 else 'retain exact dense profile rows below')
        rows.append(dict(load=c['load'],context=c['layer'],tokens=t,weight=c['weight'],operators=operators,
                         g0_dense_profile=profiles[0],g607_dense_profile=profiles[1]))
    contracts=[
      dict(id='D1',component='QKV/O/pooler/unpooler stage outputs',required='Round reconstructed FP32 partial sums to BF16 before next nonlinear/operator stage; router FP32 vector/logits separate',baseline='baseline.py274,310,314-328,340-349',delta='route/boundary cast additions; GLE_T/B1X disabled because moving rounds through linear rewrite is inequivalent',minimal_arm='retain casts/disable deferred rewrite or prove a different bit-boundary-preserving implementation'),
      dict(id='D2',component='compressed ordinary SK2',required='Preserve BF16 weights/operands and FP32 correction/reduction; producer/reader ordering',delta='only2 conditional global proxy fences plus PDL launchattribute0',minimal_arm='math may remain exact upstream; common host/progress/PDL contract cannot be omitted'),
      dict(id='D3',component='rs128 internal split partials',required='Weight/input precision and cross-stage BF16 edge explicit; exact allowed internal BF16 partial reduction unresolved',delta='entire RQ tail cut/stub; conservative internal-reduction exclusion',minimal_arm='not safely reenabled by deleting ban: resolve precision AND KS selfzero poll/ordinary launch admission separately'),
      dict(id='D4',component='Lt splitK guard',required='BF16 operands/outputs and documented intended reduction precision; splitK alone not prohibited',delta='guard accepts only0/1; broader than necessary for verified FP32-reduction plans',minimal_arm='all8 existing plans accepted in607; no selected-plan change from removing guard'),
      dict(id='D6',component='shared pre-expert effective gamma',required='Pinned baseline precomputes1+raw in BF16parameterdtype then passes to MuseSpark1xRMSNorm forward; interface parity dependency, not universal bitwise-intermediate rule',delta='none: bothG0/607 store raw g_preexp and active routes add1 inFP32',minimal_arm='resolve once as a common source epoch; do not charge as integration regression'),
      dict(id='D5',component='short FP32 router',required='Use unrounded FP32 normalized router vector from pinned JIT norm boundary; BF16 pooler input remains distinct',delta='GLE_FRT retained; GLE_T/B1X removed for different boundary issues',minimal_arm='do not replace hr with BF16-upcast h or feed FP32 hr to pooler'),
    ]
    return dict(root_read_head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),requested_reconciliation_commit='0f51f77',
                source_pins=[dict(path=str(p),bytes=len(b),sha256=sha(b))for p,b in zip(paths,raw)],unchanged_config_bindings=equals,unchanged_function_ast=funcs,cells=rows,contracts=contracts,
                host_flags=dict(ordinary_bfp=['-O3','nativeH200 sm_90a','no use_fast_math','optional SK2_DELAY_ESC only when nonzero; no such selected profile evidence'],
                    gle_frt=['-O3','compute_90/sm_90','no use_fast_math'],rq=['-O3','sm_90a','-DNDEBUG','--expt-relaxed-constexpr'],
                    global_pdl=[env[0]['PDL'],env[1]['PDL']],bfp_pdl_policy='_bfp_pdl(T,role) returns0 when PDLFalse; proxy fences conditional on pdl',
                    Lt='compute32F/BF16operands; fullfeaturebatch layouts unchanged; nonsplit guard added; all8 accepted'),
                minimal_arm=dict(generated=False,conforming_claim=False,existing_tools_sufficient_without_refactor=False,
                    reason='Existing fork tools couple mandatory stage casts with conservative RQ removal/Lt guard and common ordering/ownership repairs. Source generation is possible, safe mandatory-only RQ restoration is not established.',
                    common_dependencies=['current native expert progress/precision epoch','required attention BF16 value boundary','new common ATT7 cooperative progress repair/proof; inherited issue notintegrationdelta','shared preexpert effective BF16gamma interface parity','per-state scratch/prefetch/Ltworkspace ownership','PDL crossproxy handoffs if enabled'],
                    no_rq_option='G0 projection math with mandatory downstream BF16 casts, RQ excluded, current accepted Lt plans retained reproduces607 dense policy; no distinct measured treatment',
                    rq_option='Requires separate precision decision, per-tile KS selfzero progress/admission and all writer/async fences, exact prepare/tuner/pack probes; no variant emitted'),
                GPU=False,performance_forecast=None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();d=audit(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
    print('DENSE_INVENTORY_OK',len(d['cells']),'cells',len(d['contracts']),'contracts','variant_generated=False')
