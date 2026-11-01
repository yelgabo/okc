"""CHECK ONLY fixedcached_band32. Private captures; untouched policy outside16/32."""


def _bc_check(out,ref,limit=.008):
    import math
    a,b=out.float(),ref.float()
    assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),'CACHED_BAND32_NONFINITE'
    diff=a-b;rn=b.norm(dim=-1);floor=(.1*rn.mean()).clamp_min(1.e-12)
    g=float(diff.norm()/b.norm().clamp_min(1.e-12));r=float((diff.norm(dim=-1)/torch.maximum(rn,floor)).max())
    assert math.isfinite(g) and math.isfinite(r) and g<=limit and r<=.075,('CACHED_BAND32_MATH',g,r,limit)
    return g,r


def _bc_summary(records):
    import math
    assert len(records)==18 and {(m,s) for m,s,_ in records}=={(m,s) for m in (1,8,16,32,128,256) for s in (993727,993733,993739)}
    assert all(len(t)==6 and all(math.isfinite(v) and v>0 for v in t) for m,s,t in records),'CACHED_BAND32_BAD_TIMING'
    pair=[];nulls=[];regress=False
    crosses=((0,2),(0,3),(1,2),(1,3))
    for base,test in crosses:
        logs=[]
        for m in (1,8,16,32,128,256):
            rows=[r[2] for r in records if r[0]==m]
            means=[sum(row[i] for row in rows)/len(rows) for i in range(6)]
            logs.append(math.log(means[base]/means[test]))
        pair.append(math.exp(sum(logs)/6))
    for m,seed,t in records:
        nulls.append(max(abs(math.log(t[0]/t[1])),abs(math.log(t[2]/t[3]))))
        regress|=max(t[2:4])-min(t[:2])>max(.5,.01*min(t[:2]))
    null=math.exp(sum(nulls)/len(nulls))-1
    seeds=[];seed_gate=True
    for seed in (993727,993733,993739):
        rows=[r[2] for r in records if r[1]==seed]
        ratios=[math.exp(sum(math.log(t[a]/t[b]) for t in rows)/6) for a,b in crosses]
        sn=math.exp(sum(max(abs(math.log(t[0]/t[1])),abs(math.log(t[2]/t[3]))) for t in rows)/6)-1
        seed_gate &= min(ratios)>=1.01 and min(ratios)-1>2*sn
        seeds.append((seed,ratios,sn))
    keep=min(pair)>=1.01 and min(pair)-1>2*null and seed_gate and not regress
    def factors(crosses):
        return [math.exp(sum(math.log(sum(r[2][a] for r in records if r[0]==m)/sum(r[2][b] for r in records if r[0]==m)) for m in (1,8,16,32,128,256))/6) for a,b in crosses]
    direct_null=math.exp(sum(abs(math.log(t[4]/t[5])) for m,s,t in records)/len(records))-1
    return dict(cross=pair,null=null,seeds=seeds,seed_gate=seed_gate,regression=regress,keep=keep,
                direct_descriptive=dict(vs_control=factors(((0,4),(0,5),(1,4),(1,5))),
                cached_vs_direct=factors(((4,2),(4,3),(5,2),(5,3))),duplicate_null=direct_null))


def _bc_prepare_impl(weights):
    import concurrent.futures
    import statistics
    began=time.monotonic()
    assert weights['w13'].shape[0]==weights['w2'].shape[0]==256
    torch.backends.cuda.matmul.allow_tf32=False
    dev=weights['w13'].device
    namespaces=[]
    for i in range(6):
        ns={'__name__':'owned_bc_arm'+str(i),'_MD_MODULE_NAME':'owned_bc_993727_arm'+str(i)}
        exec(_BC_CONTROL if i<2 else _BC_CANDIDATE if i<4 else _BC_DIRECT,ns)
        ns['SKINNY_VARIANTS']=((8,3,0),);ns['MAIN_MODES']=(47,);ns['VARIANT_MAIN0']=1
        namespaces.append(ns)
    # Only CPU extension build/import overlaps. No tuner, GPU packing/reference
    # work or process-global math setting runs concurrently with this pool.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        builds=list(pool.map(lambda ns:ns['_ext'](),(namespaces[0],namespaces[2],namespaces[4])))
    modules=[builds[0],builds[0],builds[1],builds[1],builds[2],builds[2]]
    for ns,module in zip(namespaces,modules):ns['_EXT']=module
    torch.cuda.synchronize()
    states=[];resources=[]
    for ns,ext in zip(namespaces,modules):
        assert time.monotonic()-began<220,'CACHED_BAND32_BUILD_BUDGET'
        # Common baseline-parity repair; preserve float* kernel ABI while
        # materializing the baseline's effective BF16 parameter in prepare.
        st={'w':ns['_repack'](weights,0,32),'nw':(weights['post_norm_w'].float()+1).bfloat16().float().contiguous(),
            'scratch':ns['_scratch'](dev)}
        states.append(st)
        torch.cuda.synchronize()
        resources.append(ext.multiply_resources())
    assert all(resources[i]==resources[i+1] for i in (0,2,4)),('CACHED_BAND32_DUP_RESOURCE',resources)
    for offset in (0,8):
        a,b,d=[resources[i][offset:offset+8] for i in (0,2,4)]
        assert len(a)==len(b)==len(d)==8 and a[5]>=1,('CACHED_BAND32_RESOURCE_SHAPE',resources)
        if offset==8:
            assert a==b==d,('CACHED_BAND32_UNCHANGED_MAIN_RESOURCE',resources)
        else:
            assert a==d,('CACHED_BAND32_DIRECT_RESOURCE',resources)
            assert a[0]==b[0] and b[1]<=a[1] and b[2]<=a[2] and a[3]==b[3] and b[4]==a[4]+1408 and b[5]>=a[5] and a[6:]==b[6:],('CACHED_BAND32_RESOURCE',resources)
    assert len({s['w']['w13'].data_ptr() for s in states})==6
    assert len({s['scratch']['ctr'].data_ptr() for s in states})==6
    mutable=[v.untyped_storage().data_ptr() for st in states for v in st['scratch'].values()]
    assert len(set(mutable))==len(mutable),'CACHED_BAND32_SCRATCH_ALIAS'
    # Packing is unchanged. Every packed tensor is byte-equal in all six arms.
    assert all(set(st['w'])==set(states[0]['w']) for st in states)
    for key in states[0]['w']:
        assert all(torch.equal(states[0]['w'][key],st['w'][key]) for st in states[1:]),('CACHED_BAND32_PACK_BYTES',key)
    def guard():assert time.monotonic()-began<275,'CACHED_BAND32_PREPARE_BUDGET'
    def run(i,data):
        m=data['hidden_states'].shape[0]
        # Only16/32 execute a treatment kernel; all other computing arms use frozen control.
        owner=i if m in (16,32) else i%2
        return namespaces[owner]['_run'](data,states[i],0 if m<=32 else 1)
    def reference(data):return namespaces[0]['_ref_forward'](data,weights)
    def spin():
        assert all(int(s['scratch']['ctr'][8])==0 for s in states),'CACHED_BAND32_SPIN'
    flush=torch.empty(64*1024*1024,dtype=torch.int32,device=dev)
    records=[];checks=dict(normal=0,fresh=0,skew=0,empty=0,post=0);worst=[0.,0.]
    for m in (1,8,16,32,128,256):
        for seed in (993727,993733,993739):
            guard()
            data=namespaces[0]['_make_inputs'](m,1009*seed+m,'normal')
            ref=reference(data)
            inputs=[dict(data,**{k:data[k].clone() for k in ('hidden_states','topk_ids','topk_weights')}) for _ in range(6)]
            mutable=[v.untyped_storage().data_ptr() for inp in inputs for k,v in inp.items() if k in ('hidden_states','topk_ids','topk_weights')]
            assert len(set(mutable))==18,'CACHED_BAND32_INPUT_ALIAS'
            graphs=[];outputs=[]
            for i in range(6):
                guard()
                run(i,inputs[i])
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out=run(i,inputs[i])
                out.fill_(float('nan'))
                for _ in range(3):graph.replay()
                graph._band_inputs=inputs[i]
                graphs.append(graph);outputs.append(out)
            torch.cuda.synchronize();spin()
            for out in outputs:
                err=_bc_check(out,ref);checks['normal']+=1;worst=[max(worst[j],err[j]) for j in range(2)]
            fresh=namespaces[0]['_make_inputs'](m,2017*seed+m,'normal');fresh_ref=reference(fresh)
            for inp in inputs:
                for k in ('hidden_states','topk_ids','topk_weights'):inp[k].copy_(fresh[k])
            for out in outputs:out.fill_(float('nan'))
            for _ in range(32):
                guard()
                for graph in graphs:graph.replay()
            torch.cuda.synchronize();spin()
            for out in outputs:
                err=_bc_check(out,fresh_ref);checks['fresh']+=1;worst=[max(worst[j],err[j]) for j in range(2)]
            if seed==993727:
                skew=namespaces[0]['_make_inputs'](m,2027+m,'skewed');skew_ref=reference(skew)
                for inp in inputs:
                    for k in ('hidden_states','topk_ids','topk_weights'):inp[k].copy_(skew[k])
                for out in outputs:out.fill_(float('nan'))
                guard()
                for graph in graphs:graph.replay()
                torch.cuda.synchronize();spin()
                for out in outputs:
                    err=_bc_check(out,skew_ref);checks['skew']+=1;worst=[max(worst[j],err[j]) for j in range(2)]
                for inp in inputs:inp['topk_ids'].copy_(torch.arange(32,40,device=dev,dtype=torch.int32)[None,:].expand(m,8))
                for graph,out in zip(graphs,outputs):out.fill_(17);graph.replay()
                torch.cuda.synchronize();spin()
                for out in outputs:
                    assert out.dtype==torch.bfloat16 and bool(torch.isfinite(out).all()) and int(torch.count_nonzero(out))==0,'CACHED_BAND32_EMPTY'
                    checks['empty']+=1
            for inp in inputs:
                for k in ('hidden_states','topk_ids','topk_weights'):inp[k].copy_(data[k])
            for graph in graphs:graph.replay()
            torch.cuda.synchronize();spin()
            for out in outputs:_bc_check(out,ref)
            events=[[] for _ in range(6)]
            for rep in range(20):
                guard()
                order=list(range(6));shift=(rep+seed)%6;order=order[shift:]+order[:shift]
                if rep%2:order.reverse()
                for i in order:
                    guard()
                    flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                    a.record();graphs[i].replay();z.record();z.synchronize();events[i].append((a,z))
            torch.cuda.synchronize();spin()
            for out in outputs:
                err=_bc_check(out,ref);checks['post']+=1;worst=[max(worst[j],err[j]) for j in range(2)]
            t=[statistics.median(a.elapsed_time(z)*1000 for a,z in pairs) for pairs in events]
            records.append((m,seed,t))

    guard()
    assert checks==dict(normal=108,fresh=108,skew=36,empty=36,post=108),checks
    import json
    report=dict(schema='OWNED_CACHED_BAND32_SIXARM_V1',source=_BAND_SOURCE_MANIFEST,
                checks=checks,worst=worst,resources=resources,records=records,
                gate=_bc_summary(records),treatment_m=[16,32],seconds=time.monotonic()-began)
    text=json.dumps(report,separators=(',',':'),allow_nan=False)
    assert len(text.encode())+100<11000,'CACHED_BAND32_LOG_BUDGET'
    guard()
    print('CACHED_BAND32_REPORT',text,flush=True)
    print('CACHED_BAND32_DONE 18cases_complete CHECK_ONLY no_submit',flush=True)
    return {'namespaces':namespaces,'states':states}


def prepare(weights):
    saved_tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        return _bc_prepare_impl(weights)
    finally:
        torch.backends.cuda.matmul.allow_tf32=saved_tf32


def kernel_fn(data,state):
    # Judged control follows exact published shape policy; no candidate selection.
    ns=state['namespaces'][0]
    return ns['_run'](data,state['states'][0],0 if data['hidden_states'].shape[0]<=32 else 1)
