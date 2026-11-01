"""Prepare-only qualification; no comparative benchmark or timing selection."""
BODY=r'''
_SXR_NORMAL_PREPARE=prepare

def _sxr_math(out,ref,x,candidate=False):
    assert bool(torch.isfinite(ref).all()),'SXRELEASE_REFERENCE_NONFINITE'
    metrics=_sx_check(out,ref,x)
    assert all(__import__('math').isfinite(v) for v in metrics),'SXRELEASE_METRIC_NONFINITE'
    assert metrics[0]<=SELFTEST_TOL['bf16'] and metrics[1]<=5*SELFTEST_TOL['bf16'],'SXRELEASE_C2_MATH_GATE'
    probes=(x!=0).sum(1)==1
    if bool(probes.any()):
        pe=((out[probes].double()-ref[probes].double()).norm(dim=1)/ref[probes].double().norm(dim=1).clamp_min(1e-30)).max()
        assert float(pe)<=PROBE_TOL,'SXRELEASE_C2_PROBE_GATE'
        if candidate:assert torch.equal(out[probes],ref[probes]),'SXRELEASE_EXACT_PROBE'
    return metrics

def _sxr_resource(name,info,control):
    r=info['resources'];expected=_sx_layout(name,info['variant'],info['pack']['pmax'])
    assert len(r)==7 and r[1]==0 and r[6]>=1,('SXRELEASE_RESOURCE',name,r)
    assert r[2]==expected['static'] and r[3]==expected['dynamic'] and r[4]==r[5]==expected['threads'],('SXRELEASE_LAYOUT',name,r,expected)
    if control is not None:
        assert r[6]>=control['resources'][6],('SXRELEASE_ADMISSION',name,r,control['resources'])
    return list(r)

def _sxr_qualify(weights,state,start):
    _sx_guard(start)
    module=xor_build('release_sk10_e7xor1')
    _sx_guard(start)
    originals=dict(state['pick']);ready={};cache={};rows=[];diagnostics={}
    checks=0;captures=0;fresh_checks=0;qualified=0
    for index,((layer,m),name) in enumerate(_SX_TARGET.items()):
        _sx_guard(start);weight=weights[layer];packs=[]
        # Same twelve immutable formats as the audited paired harness. Failed
        # finite/sparse offer is a legal incumbent fallback, not qualification.
        for variant in (1,0):
            key=_sx_pack_key(name,weight,variant)
            if key not in cache:cache[key]=xor_pack(name,weight,variant)
            pk=cache[key]
            packs.append(None if pk is None else dict(pk,name=name))
            if pk is None:break
        if len(packs)!=2 or any(p is None for p in packs):
            rows.append(dict(layer=layer,m=m,name=name,status='unsupported_pack_original_C2',checks=0))
            continue
        candidate_pack,control_pack=packs
        control,ci=xor_runner(module,name,weight,0,packed=control_pack)
        candidate,ti=xor_runner(module,name,weight,1,packed=candidate_pack)
        control_nf,cn=xor_runner(module,name,weight,0,nearby=True,packed=control_pack)
        candidate_nf,tn=xor_runner(module,name,weight,1,nearby=True,packed=candidate_pack)
        vectors=[_sxr_resource(name,ci,None),_sxr_resource(name,ti,ci),
                 _sxr_resource(name,cn,None),_sxr_resource(name,tn,cn)]
        for label,info in zip(('control','candidate','control_nf','candidate_nf'),(ci,ti,cn,tn)):
            diagnostics[name+':'+label]=[info['private_diag']]
        _sx_disjoint(diagnostics)
        gen=torch.Generator(device=weight.device).manual_seed(914301+index*97+m)
        n=candidate_pack['n'];k=weight.shape[1];case_checks=0
        for mm,functions in ((m,(control,candidate)),(max(1,m-1),(control_nf,candidate_nf))):
            for bank in ('normal','onehot','outlier'):
                _sx_guard(start)
                x=_sx_x(mm,k,gen,weight.device,onehot=bank=='onehot',outlier=bank=='outlier')
                ref=_sx_reference(x,weight,n)
                for variant,fn in enumerate(functions):
                    out=fn(x);torch.cuda.synchronize()
                    _sxr_math(out,ref,x,bool(variant));case_checks+=1
                del x,ref,out
        # Only the single treatment is captured here. No events/latency samples.
        x=_sx_x(m,k,gen,weight.device);ref=_sx_reference(x,weight,n)
        graph,out,stream=_sx_capture(candidate,x);captures+=1
        _sxr_math(out,ref,x,True);case_checks+=1
        for replay in range(3):
            _sx_guard(start)
            fresh=_sx_x(m,k,gen,weight.device)
            ref=_sx_reference(fresh,weight,n)
            x.copy_(fresh);out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
            _sxr_math(out,ref,x,True);case_checks+=1;fresh_checks+=1
        for info in (ci,ti,cn,tn):
            assert int(torch.count_nonzero(info['private_diag']))==0,('SXRELEASE_DEVICE_DIAGNOSTIC',name)
        assert case_checks==16,('SXRELEASE_CASE_COVERAGE',name,case_checks)
        checks+=case_checks;qualified+=1
        first=2 if m==8 else 9 if m==16 else 17
        for nearby_m in range(first,m):ready[layer,nearby_m]=candidate_nf
        ready[layer,m]=candidate
        rows.append(dict(layer=layer,m=m,name=name,status='override',checks=case_checks,
            installed_range=[first,m],resources=vectors,register_thread_products=[r[0]*r[4] for r in vectors]))
        del graph,out,stream,x,ref,fresh,control,control_nf,candidate_nf,ci,ti,cn,tn
        _sx_guard(start)
    allowed={(layer,mm) for layer,m in _SX_TARGET for mm in range(2 if m==8 else 9 if m==16 else 17,m+1)}
    assert len(rows)==10 and set(ready).issubset(allowed),'SXRELEASE_COVERAGE'
    assert checks==16*qualified and captures==qualified and fresh_checks==3*qualified,'SXRELEASE_INCOMPLETE'
    for buffers in diagnostics.values():
        for diag in buffers:assert int(torch.count_nonzero(diag))==0,'SXRELEASE_FINAL_DIAGNOSTIC'
    _sx_guard(start)
    # Atomic host publication after every required check has passed. Untargeted
    # keys, pick_base, general-M fallbacks and normal kernel_fn stay incumbent.
    assert state['pick']==originals,'SXRELEASE_INCUMBENT_MUTATED'
    state['pick'].update(ready)
    assert all(state['pick'][key] is fn for key,fn in originals.items() if key not in ready),'SXRELEASE_UNTARGETED_MUTATED'
    state['xor_release']={'source':_SXR_PINS,'overrides':qualified,'fallbacks':10-qualified,
        'installed_pick_keys':len(ready),'nearby_pick_keys':len(ready)-qualified,
        'math_checks':checks,'captures':captures,'fresh_checks':fresh_checks,'routes':rows}
    _sx_guard(start)
    for row in rows:
        print('SXRELEASE_ROUTE',row['layer'],row['m'],row['status'],'checks='+str(row['checks']),
              'resources='+str(row.get('resources',[])),flush=True)
    print('SXRELEASE_DONE overrides='+str(qualified)+' fallbacks='+str(10-qualified)+
          ' pick_keys='+str(len(ready))+' checks='+str(checks)+' captures='+str(captures)+' fresh='+str(fresh_checks),flush=True)
    _sx_guard(start)
    # Only immutable weight packs, selected runners and their diagnostics survive.
    return state

def prepare(weights):
    start=time.monotonic()
    state=_SXR_NORMAL_PREPARE(weights)
    _sx_guard(start)
    return _sxr_qualify(weights,state,start)
'''
