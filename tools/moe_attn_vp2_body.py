"""Import-time private four-owner comparator; all exceptions are fatal."""
_VP_BEGIN=time.monotonic()

def _vp_guard():
    assert time.monotonic()-_VP_BEGIN<275,'VP2_PRIVATE_BUDGET'

def _vp_clone(data):
    return {k:v.clone() if isinstance(v,torch.Tensor) else dict(v) if isinstance(v,dict) else v for k,v in data.items()}

def _vp_refs(data):
    c=data['config'];b,l,h=c['batch'],c['q_len'],c['num_kv_heads']
    _,ki,vi,_=data['qkv'].split([c['num_q_heads']*64,h*64,h*64,c['num_q_heads']*64],-1)
    pos=data['seq_lens'].long()[:,None]+torch.arange(l,device=ki.device)
    k=_ob_rms(ki.reshape(b,l,h,64))
    if c['rope_theta']:k=_ob_rope(k,pos)
    kn=(k/data['k_scale']).to(FP8);vn=(vi.reshape(b,l,h,64).float()/data['v_scale']).to(FP8)
    raw=_ob_reference(data,False,(kn,vn)).bfloat16()
    e4=_ob_reference(data,True,(kn,vn)).bfloat16()
    bf=_vp_bf_reference(data,False,(kn,vn)).bfloat16()
    assert all(bool(torch.isfinite(v.float()).all()) for v in (raw,e4,bf,kn,vn)),'VP2_REFERENCE_NONFINITE'
    bound=min(.15,max(.03,2.5*_ob_rel(e4,raw)+.015))
    return raw,bf,kn,vn,bound

def _vp_check(data,out,refs):
    raw,bf,kn,vn,bound=refs;c=data['config'];h=c['num_kv_heads'];l=c['q_len']
    assert out.dtype==torch.bfloat16 and bool(torch.isfinite(out.float()).all()),'VP2_OUTPUT_NONFINITE'
    errors=[]
    for ref in (raw,bf):
        assert bool(torch.isfinite(ref.float()).all()),'VP2_REFERENCE_NONFINITE'
        a=out.float();b=ref.float();rn=b.reshape(-1,64).norm(dim=-1)
        floor=(.1*rn.mean()).clamp_min(1e-12)
        ge=float((a-b).norm()/b.norm().clamp_min(1e-12))
        re=float(((a-b).reshape(-1,64).norm(dim=-1)/torch.maximum(rn,floor)).max())
        assert math.isfinite(ge) and math.isfinite(re) and ge<=bound and re<=5*bound,('VP2_MATH',ge,re,bound)
        errors.extend((ge,re))
    pos=data['seq_lens'].long()[:,None]+torch.arange(l,device=out.device)
    slots=torch.gather(data['page_table'].long(),1,pos//c['page_size'])*c['page_size']+pos%c['page_size']
    kg=data['k_cache'].view(-1,h,64)[slots].float();vg=data['v_cache'].view(-1,h,64)[slots].float()
    ek=_ob_rel(kg,kn.float());ev=_ob_rel(vg,vn.float())
    assert math.isfinite(ek) and math.isfinite(ev) and ek<=.08 and ev<=.08,('VP2_CACHE',ek,ev)
    return errors+[ek,ev]

def _vp_score(rows):
    assert len(rows)==36 and {(b,l,w,s) for b,l,w,s,t in rows}=={(b,l,w,s) for b in (1,16,32) for l in (1,8) for w in (0,1) for s in _VP_MANIFEST['seeds']}
    assert all(len(t)==4 and all(math.isfinite(v) and v>0 for v in t) for *_,t in rows),'VP2_BAD_TIMING'
    cross=((0,2),(0,3),(1,2),(1,3));per=[];pooled=[];regression=False
    def ratio(selected,a,b):
        logs=[]
        for batch in (1,16,32):
            for length in (1,8):
                mix=[]
                for arm in (a,b):
                    mix.append(sum((46 if win else 16)*statistics.mean(r[4][arm] for r in selected if tuple(r[:3])==(batch,length,win)) for win in (0,1)))
                assert min(mix)>0,'VP2_CALIBRATION_NONPOSITIVE'
                logs.append(math.log(mix[0]/mix[1]))
        return math.exp(sum(logs)/6)
    for seed in _VP_MANIFEST['seeds']:
        selected=[r for r in rows if r[3]==seed]
        rs=[ratio(selected,a,b) for a,b in cross]
        null=max(abs(ratio(selected,0,1)-1),abs(ratio(selected,2,3)-1))
        per.append([seed,rs,null,min(rs)>=1.01 and min(rs)-1>2*null])
    for b,l,w,s,t in rows:
        regression|=max(t[2:])-min(t[:2])>max(.5,.01*min(t[:2]))
    pooled=[ratio(rows,a,b) for a,b in cross]
    null=max(abs(ratio(rows,0,1)-1),abs(ratio(rows,2,3)-1))
    return dict(cross=pooled,null=null,per_seed=per,regression=regression,
                keep=min(pooled)>=1.01 and min(pooled)-1>2*null and all(r[-1] for r in per) and not regression)

def _vp_prepare():
    from torch.utils.cpp_extension import load_inline
    prior=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        ns=[]
        for i in range(4):
            n={'__name__':'vp_owned_owner'+str(i)};exec(_VP_OWNED,n);ns.append(n)
            assert n['_ONE'] is None and n['_ONE_WS_PICK']==0 and not n['_ONE_STATE']
        def build(i):
            if i<2:
                n=ns[i]
                return load_inline(name='vp2bf_owned824731_'+str(i),cpp_sources=n['_ONE_CPP'],cuda_sources=n['_ONE_SRC'],
                    functions=['attn_one','attn_one_prepare','attn_one_probe_coop','vp_owned_resources'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'])
            return load_inline(name='vp2bf_foreign824731_'+str(i),cpp_sources=_VP_CPP,cuda_sources=_VP_CUDA,
                functions=['muse_vp2_launch','vp2_resources'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a',
                '-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__','-DZH_SCH=0'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:mods=list(pool.map(build,range(4)))
        _vp_guard();torch.cuda.synchronize()
        for i,n in enumerate(ns):n['_ONE']=mods[i%2]
        resources=[list(m.vp_owned_resources()) if i<2 else list(m.vp2_resources()) for i,m in enumerate(mods)]
        assert resources[0]==resources[1] and resources[2]==resources[3],'VP2_DUP_RESOURCE'
        assert all(resources[i][-1]>=2 and resources[i][1]==0 for i in (2,3))
        dev=torch.device('cuda',torch.cuda.current_device())
        for n in ns:n['_one_state'](dev)
        ptrs=[v.untyped_storage().data_ptr() for n in ns for k,v in n['_ONE_STATE'][dev].items() if isinstance(v,torch.Tensor)]
        assert len(ptrs)==len(set(ptrs)),'VP2_STATE_ALIAS'
        plans=[]
        dispatch_counts={'ONE':0,'VP2':0}
        def eligible(c):
            return(c['batch']==1 and c['q_len']==8 and c['window']==2048 and c['page_size']==1 and
                   c['kv_scale']==1 and c['num_q_heads']==16 and c['num_kv_heads']==2)
        def run(i,data,record=True):
            c=data['config']
            if not ns[i]['_one_ok'](c):
                assert not record,'VP2_NATIVE_CONTROL_REQUIRED'
                return ns[i]['kernel_fn'](data)
            if i<2 or not eligible(c):
                if record:dispatch_counts['ONE']+=1
                return ns[i]['_one'](data)
            if record:dispatch_counts['VP2']+=1
            out=torch.empty((1,8,1024),device=data['qkv'].device,dtype=torch.bfloat16)
            q=data['qkv'];p=data['page_table'];freq=ns[i]['_inv_freq'](c['rope_theta'],64,q.device)
            mods[i].muse_vp2_launch(q.data_ptr(),p.data_ptr(),p.stride(0),data['seq_lens'].data_ptr(),data['cache_loc'].data_ptr(),
                data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),freq.data_ptr(),out.data_ptr(),1,2048,
                c['qk_scale_factor']*c['sm_scale']*ns[i]['LOG2E'],1.,1.,1.,1e-5,0,torch.cuda.current_stream().cuda_stream)
            return out
        flush=torch.empty(256<<20,device='cuda',dtype=torch.uint8)
        rows=[];raw_rows=[];counts=dict(eager_extra=0,normal=0,fresh=0,post=0);worst=[0.]*6
        # Same target ABI, short/empty source partitions; never contributes timing credit.
        for context in (0,10,2047,2048):
            _vp_guard();base=_ob_inputs(1,context,'tp8','sliding',8,'fixed','soft',1,1.,seed=824719+context)
            refs=_vp_refs(base)
            for i in range(4):
                data=_vp_clone(base);out=run(i,data);torch.cuda.synchronize()
                e=_vp_check(data,out,refs);worst=[max(a,b) for a,b in zip(worst,e)];counts['eager_extra']+=1
        for batch in (1,16,32):
            for length in (1,8):
                for window in (0,1):
                    for seed in _VP_MANIFEST['seeds']:
                        _vp_guard();layer='sliding' if window else 'full'
                        base=_ob_inputs(batch,80000,'tp8',layer,length,'fixed','soft',1,1.,seed=seed+batch*17+length*71+window)
                        refs=_vp_refs(base);inputs=[_vp_clone(base) for _ in range(4)]
                        cacheptr=[d[k].untyped_storage().data_ptr() for d in inputs for k in ('qkv','k_cache','v_cache')]
                        assert len(set(cacheptr))==12,'VP2_INPUT_ALIAS'
                        for i in range(4):
                            st=ns[i]['_ONE_STATE'][dev]
                            plan=ns[i]['_one_plan'](base['config'],base['page_table'].shape[1],st['resident'])
                            assert (length,plan[2]) in st['resident'] and st['resident'][(length,plan[2])]>0,'VP2_NATIVE_PLAN'
                            rr=resources[i%2];selected=next(rr[q:q+7] for q in range(0,len(rr),7) if rr[q:q+2]==[length,plan[2]])
                            assert selected[3]==0 and selected[6]>0,('VP2_NATIVE_RESOURCE',selected)
                            if seed==_VP_MANIFEST['seeds'][0]:
                                actual=['VP2_grid_cluster_threads',32,16,256] if i>=2 and eligible(base['config']) else ['ONE_splits_copies_mode',*plan]
                                plans.append([batch,length,window,i,actual])
                        graphs=[];outputs=[]
                        for i in range(4):
                            run(i,inputs[i]);g=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g):out=run(i,inputs[i])
                            g.replay();graphs.append(g);outputs.append(out)
                        torch.cuda.synchronize()
                        for d,o in zip(inputs,outputs):
                            e=_vp_check(d,o,refs);worst=[max(a,b) for a,b in zip(worst,e)];counts['normal']+=1
                        original=base['qkv'].clone()
                        fresh_count=17 if batch==1 and length==8 and window else 1
                        generator=torch.Generator(device='cuda').manual_seed(seed+99317)
                        for fresh in range(fresh_count):
                            _vp_guard();base['qkv']=torch.randn(original.shape,device='cuda',generator=generator).bfloat16()
                            freshrefs=_vp_refs(base)
                            for d in inputs:d['qkv'].copy_(base['qkv'])
                            for g in graphs:g.replay()
                            torch.cuda.synchronize()
                            for d,o in zip(inputs,outputs):
                                e=_vp_check(d,o,freshrefs);worst=[max(a,b) for a,b in zip(worst,e)];counts['fresh']+=1
                        for d in inputs:d['qkv'].copy_(original)
                        for g in graphs:g.replay()
                        torch.cuda.synchronize()
                        events=[[] for _ in range(4)]
                        for rep in range(32):
                            _vp_guard();order=list(range(4));shift=(rep+seed)%4;order=order[shift:]+order[:shift]
                            if rep%2:order.reverse()
                            for i in order:
                                _vp_guard();flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                                a.record();graphs[i].replay();z.record();z.synchronize();events[i].append((a,z))
                        for d,o in zip(inputs,outputs):
                            e=_vp_check(d,o,refs);worst=[max(a,b) for a,b in zip(worst,e)];counts['post']+=1
                        raw=[statistics.median(a.elapsed_time(z)*1000 for a,z in pairs) for pairs in events]
                        offset=_VP_OFFSETS[(batch,length,window)]
                        adjusted=[v-offset for v in raw];assert min(adjusted)>0,'VP2_BAD_CALIBRATION'
                        rows.append((batch,length,window,seed,adjusted))
                        raw_rows.append((batch,length,window,seed,raw))
                        del graphs,outputs,inputs,base,refs
        _vp_guard();assert counts==dict(eager_extra=16,normal=144,fresh=336,post=144),counts
        # Count host warm/capture calls, not graph replays. An all-fallback run cannot qualify.
        assert dispatch_counts=={'ONE':284,'VP2':20},('VP2_DISPATCH_COVERAGE',dispatch_counts)
        report=dict(schema='PRIVATE_VP2_BF16_V1',source=_VP_MANIFEST,resources=resources,plans=plans,
                    counts=counts,dispatch_counts=dispatch_counts,worst=worst,rows=raw_rows,row_kind='raw_us',
                    fixed_offsets=[[b,l,w,v] for (b,l,w),v in _VP_OFFSETS.items()],
                    gate=_vp_score(rows),raw_gate=_vp_score(raw_rows),seconds=time.monotonic()-_VP_BEGIN,
                    calibration='gate uses fixed earlier3f44 offsets, NOT decision alone; root must recompute paired sensitivity using contemporaneous official offsets; raw_gate unadjusted')
        text=__import__('json').dumps(report,separators=(',',':'),allow_nan=False)
        assert len(text.encode())<11000,'VP2_LOG_BUDGET';_vp_guard()
        print('VP2_PRIVATE_REPORT',text,flush=True);print('VP2_PRIVATE_DONE 36 private_only',flush=True)
        return ns,mods,run
    finally:torch.backends.cuda.matmul.allow_tf32=prior

_VP_NS,_VP_MODS,_VP_RUN=_vp_prepare()

def kernel_fn(data):
    # PRIVATE comparator is judged on VP2 only where eligible; owned/native or stock otherwise.
    return _VP_RUN(2,data,record=False)
