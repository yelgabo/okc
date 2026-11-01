"""Prepare-only qualification; no comparative benchmark or timing selection."""
BODY=r'''
_SXSR_NORMAL_PREPARE=prepare

def _sxsr_math(out,ref,x,candidate=False):
    assert bool(torch.isfinite(ref).all()),'SXSRELEASE_REFERENCE_NONFINITE'
    metrics=_sx_check(out,ref,x)
    assert all(__import__('math').isfinite(v) for v in metrics),'SXSRELEASE_METRIC_NONFINITE'
    assert metrics[0]<=SELFTEST_TOL['bf16'] and metrics[1]<=5*SELFTEST_TOL['bf16'],'SXSRELEASE_C2_MATH_GATE'
    probes=(x!=0).sum(1)==1
    if bool(probes.any()):
        pe=((out[probes].double()-ref[probes].double()).norm(dim=1)/ref[probes].double().norm(dim=1).clamp_min(1e-30)).max()
        assert float(pe)<=PROBE_TOL,'SXSRELEASE_C2_PROBE_GATE'
        if candidate:assert torch.equal(out[probes],ref[probes]),'SXSRELEASE_EXACT_PROBE'
    return metrics

def _sxsr_resource(name,info,control):
    r=info['resources'];expected=sxs_expected_layout(name,info['variant'],info['pack']['pmax'])
    assert len(r)==7 and r[1]==0 and r[6]>=1,('SXSRELEASE_RESOURCE',name,r)
    assert expected['dynamic']<=227*1024,('SXSRELEASE_DYNAMIC_EXTENT',name,expected)
    assert r[2]==expected['static'] and r[3]==expected['dynamic'] and r[4]==r[5]==expected['threads'],('SXSRELEASE_LAYOUT',name,r,expected)
    if control is not None:
        assert r[6]>=control['resources'][6],('SXSRELEASE_ADMISSION',name,r,control['resources'])
    return list(r)

def _sxsr_qualify(weights,state,start):
    _sx_guard(start)
    module=sxs_build('release_sk10_stage_e7xor1')
    _sx_guard(start)
    originals=dict(state['pick']);ready={};cache={};rows=[];diagnostics={}
    checks=0;captures=0;fresh_checks=0;qualified=0
    for index,((layer,m),name) in enumerate(_SX_TARGET.items()):
        _sx_guard(start);weight=weights[layer];packs=[]
        # Same twelve immutable formats as the audited paired harness. Failed
        # finite/sparse offer is a legal incumbent fallback, not qualification.
        for variant in (2,0):
            key=_sx_pack_key(name,weight,variant)
            if key not in cache:cache[key]=sxs_pack(name,weight,variant)
            pk=cache[key]
            packs.append(None if pk is None else dict(pk,name=name))
            if pk is None:break
        if len(packs)!=2 or any(p is None for p in packs):
            rows.append(dict(layer=layer,m=m,name=name,status='unsupported_pack_original_C2',checks=0))
            continue
        candidate_pack,control_pack=packs
        # Reject unsupported resource extents before any test kernel can launch.
        # Weight-representation offer rejection above alone permits fallback.
        for variant,pk in ((2,candidate_pack),(0,control_pack)):
            layout=sxs_expected_layout(name,variant,pk['pmax'])
            assert 0<=layout['dynamic']<=227*1024,('SXSRELEASE_DYNAMIC_EXTENT',name,layout)
        assert candidate_pack['table'].numel()==candidate_pack['n'],'SXSRELEASE_ROWMAP_ONLY'
        control,ci=sxs_runner(module,name,weight,0,packed=control_pack)
        candidate,ti=sxs_runner(module,name,weight,2,packed=candidate_pack)
        control_nf,cn=sxs_runner(module,name,weight,0,nearby=True,packed=control_pack)
        candidate_nf,tn=sxs_runner(module,name,weight,2,nearby=True,packed=candidate_pack)
        vectors=[_sxsr_resource(name,ci,None),_sxsr_resource(name,ti,ci),
                 _sxsr_resource(name,cn,None),_sxsr_resource(name,tn,cn)]
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
                    _sxsr_math(out,ref,x,bool(variant));case_checks+=1
                del x,ref,out
        # Only the single treatment is captured here. No events/latency samples.
        x=_sx_x(m,k,gen,weight.device);ref=_sx_reference(x,weight,n)
        graph,out,stream=_sx_capture(candidate,x);captures+=1
        _sxsr_math(out,ref,x,True);case_checks+=1
        for replay in range(3):
            _sx_guard(start)
            fresh=_sx_x(m,k,gen,weight.device)
            ref=_sx_reference(fresh,weight,n)
            x.copy_(fresh);out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
            _sxsr_math(out,ref,x,True);case_checks+=1;fresh_checks+=1
        for info in (ci,ti,cn,tn):
            assert int(torch.count_nonzero(info['private_diag']))==0,('SXSRELEASE_DEVICE_DIAGNOSTIC',name)
        assert case_checks==16,('SXSRELEASE_CASE_COVERAGE',name,case_checks)
        checks+=case_checks;qualified+=1
        first=2 if m==8 else 9 if m==16 else 17
        for nearby_m in range(first,m):ready[layer,nearby_m]=candidate_nf
        ready[layer,m]=candidate
        rows.append(dict(layer=layer,m=m,name=name,status='override',checks=case_checks,
            stage_capacity=candidate_pack['pmax'],stage_record_bytes=candidate_pack['stage_record_bytes'],
            weight_bytes=sxs_expected_layout(name,2,candidate_pack['pmax'])['weight_bytes'],
            installed_range=[first,m],resources=vectors,register_thread_products=[r[0]*r[4] for r in vectors]))
        del graph,out,stream,x,ref,fresh,control,control_nf,candidate_nf,ci,ti,cn,tn
        _sx_guard(start)
    allowed={(layer,mm) for layer,m in _SX_TARGET for mm in range(2 if m==8 else 9 if m==16 else 17,m+1)}
    assert len(rows)==10 and set(ready).issubset(allowed),'SXSRELEASE_COVERAGE'
    assert checks==16*qualified and captures==qualified and fresh_checks==3*qualified,'SXSRELEASE_INCOMPLETE'
    for buffers in diagnostics.values():
        for diag in buffers:assert int(torch.count_nonzero(diag))==0,'SXSRELEASE_FINAL_DIAGNOSTIC'
    _sx_guard(start)
    # Atomic host publication after every required check has passed. Untargeted
    # keys, pick_base, general-M fallbacks and normal kernel_fn stay incumbent.
    assert state['pick']==originals,'SXSRELEASE_INCUMBENT_MUTATED'
    state['pick'].update(ready)
    assert all(state['pick'][key] is fn for key,fn in originals.items() if key not in ready),'SXSRELEASE_UNTARGETED_MUTATED'
    state['stage_xor_release']={'source':_SXSR_PINS,'overrides':qualified,'fallbacks':10-qualified,
        'installed_pick_keys':len(ready),'nearby_pick_keys':len(ready)-qualified,
        'math_checks':checks,'captures':captures,'fresh_checks':fresh_checks,'routes':rows}
    _sx_guard(start)
    for row in rows:
        print('SXSRELEASE_ROUTE',row['layer'],row['m'],row['status'],'checks='+str(row['checks']),
              'capacity='+str(row.get('stage_capacity',0)),'arena_bytes='+str(row.get('stage_record_bytes',0)),
              'resources='+str(row.get('resources',[])),flush=True)
    print('SXSRELEASE_DONE overrides='+str(qualified)+' fallbacks='+str(10-qualified)+
          ' pick_keys='+str(len(ready))+' checks='+str(checks)+' captures='+str(captures)+' fresh='+str(fresh_checks),flush=True)
    _sx_guard(start)
    # Only immutable weight packs, selected runners and their diagnostics survive.
    return state

def prepare(weights):
    start=time.monotonic()
    state=_SXSR_NORMAL_PREPARE(weights)
    _sx_guard(start)
    return _sxsr_qualify(weights,state,start)
'''
