"""Prospective fixed-policy host setup/checker only. Public entry stays OFF; no graph qualification yet."""
RUNTIME=r'''
_SPG_ENABLED=False
_SPG_STATIC_GATE_PASSED=False
_SPG_COMPLETE_GRAPH_QUALIFIED=False
_SPG_POLICY={1:3,8:3,16:3,32:3,128:9,256:9}

def _spg_budget(start):
    assert time.monotonic()-start<295,'SPG_PREPARE_BUDGET'

def _spg_check(out,ref):
    assert out.shape==ref.shape and out.dtype==torch.bfloat16,'SPG_OUTPUT_ABI'
    o,r=out.float(),ref.float()
    assert bool(torch.isfinite(o).all()) and bool(torch.isfinite(r).all()),'SPG_NONFINITE'
    den=r.norm().clamp(min=1e-12);err=(o-r).norm()/den
    rows_r=r.norm(dim=-1);rows_d=(o-r).norm(dim=-1)
    row=(rows_d/torch.maximum(rows_r,(.1*rows_r.mean()).clamp(min=1e-12))).max()
    assert bool(torch.isfinite(err)) and bool(torch.isfinite(row)),'SPG_NONFINITE_METRIC'
    assert float(err)<=.008 and float(row)<=.075,('SPG_FP32_ERROR',float(err),float(row))
    return float(err),float(row)

def _spg_selected_selftest(state,weights,sizes,label,variant=None):
    # Same original natural/skew generators and FP32 computation; stricter acceptance only.
    worst=0.
    for bs in sizes:
        for dist in ('normal','skewed'):
            if (bs,dist) not in _REF_CACHE:
                data=_make_inputs(bs,200+bs,dist)
                _REF_CACHE[bs,dist]=(data,_ref_forward(data,weights))
            data,ref=_REF_CACHE[bs,dist]
            err,row=_spg_check(_run(data,state,variant),ref)
            worst=max(worst,err)
    torch.cuda.synchronize()
    assert int(state['scratch']['ctr'][8].item())==0,'SPG_POLL_DIAGNOSTIC'
    print('SPG_SELECTED_SELFTEST',label,'global',worst,'rows<=.075','flags0',flush=True)
    return True

def _spg_input_scope(data,state):
    c=data['config'];x=data['hidden_states'];m=x.shape[0]
    expected={'bs':m,'d_hidden':4096,'d_expert':4096,'n_experts':256,'local_start':0,'local_count':32,'top_k':8,'sharding':'ep8'}
    assert all(c.get(k)==v for k,v in expected.items()),'SPG_UNSUPPORTED_EP8_SCOPE'
    assert m>0 and x.ndim==2 and x.shape[1]==4096 and x.dtype==torch.bfloat16,'SPG_INPUT_ABI'
    assert x.device.type=='cuda' and x.device==state['w']['w13'].device,'SPG_DEVICE'
    assert data['topk_ids'].shape==(m,8) and data['topk_ids'].dtype in (torch.int32,torch.int64),'SPG_IDS_ABI'
    assert data['topk_weights'].shape==(m,8) and data['topk_weights'].is_floating_point(),'SPG_WEIGHTS_ABI'
    assert data['topk_ids'].device==x.device and data['topk_weights'].device==x.device,'SPG_INPUT_DEVICE'
    # Preserve original contiguous/int32/float32 coercions and512 chunk loop in _run.

def _spg_prepare_candidate(weights):
    """INTERNAL CPU/source prototype: not the complete future graph qualifier."""
    start=time.monotonic();old=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False;_REF_CACHE.clear()
        assert SKINNY_VARIANTS[3]==(8,3,0) and MAIN_MODES[5]==47 and VARIANT_MAIN0==4,'SPG_POLICY_TABLE_DRIFT'
        state={'w':_repack(weights,0,MAX_LOCAL),
               'nw':(weights['post_norm_w'].float()+1.).contiguous(),
               'nw_bf16':(weights['post_norm_w'].float()+1.).to(torch.bfloat16).contiguous(),
               'scratch':_scratch(weights['w13'].device),'legacy_ok':False,'main_ok':True,
               'main_bad':set(),'policy':dict(_SPG_POLICY)}
        _ext();_spg_budget(start)
        # Qualification writes private counters/scratch only; no timing/tuning selectors.
        private=dict(state);private['scratch']=_scratch(weights['w13'].device)
        assert _self_test(private,weights,(1,4,8,16,32,33,128,257,512,513),'frozen47+sk8w3'), 'SPG_SELECTED_SELFTEST_FALSE'
        _spg_budget(start);torch.cuda.synchronize()
        return state
    finally:
        _REF_CACHE.clear();torch.backends.cuda.matmul.allow_tf32=old

# Actual public entry intentionally cannot run until BOTH failed/missing gates are newly qualified.
def prepare(weights):
    assert _SPG_ENABLED,'SPG_OFF_BY_DEFAULT'
    assert _SPG_STATIC_GATE_PASSED,'SPG_STATIC_NEW_SPILLS'
    assert _SPG_COMPLETE_GRAPH_QUALIFIED,'SPG_FRESH_EMPTY_GRAPH_QUALIFICATION_MISSING'
    return _spg_prepare_candidate(weights)

def kernel_fn(data,state):
    assert _SPG_ENABLED and _SPG_STATIC_GATE_PASSED and _SPG_COMPLETE_GRAPH_QUALIFIED,'SPG_UNQUALIFIED_ENTRY'
    _spg_input_scope(data,state)
    return _run(data,state)
'''
