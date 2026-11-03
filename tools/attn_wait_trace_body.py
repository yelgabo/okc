_WAIT_FAILED=True
_WAIT_DBGS={}
_WAIT_ROWS=[]
_WAIT_RES={}

def _wait_resources():
    mods={'ctrl':_ONE,**_AB_MODS}
    assert set(mods)=={'ctrl','ctrl2','detailA','detailB'},'missing wait module'
    for name,mod in mods.items():
        r=list(mod.attn_wait_resources());r={tuple(r[i:i+2]):r[i+2:i+7] for i in range(0,len(r),7)}
        assert all(k in r for k in ((1,49),(8,51)))
        for k in ((1,49),(8,51)):
            assert r[k][1]==0 and r[k][-1]>=2,('wait spills/residency',name,k,r[k])
            if name!='ctrl':assert r[k]==_WAIT_RES['ctrl'][k],('wait geometry',name,k,r[k])
        _WAIT_RES[name]=r
    print('WAITRES',_WAIT_RES,flush=True)

_wait_parent_via=_ab_via

def _wait_dbg(name,dev):
    key=(name,str(dev))
    if key not in _WAIT_DBGS:_WAIT_DBGS[key]=torch.zeros(1024+512*56,device=dev,dtype=torch.int32)
    return _WAIT_DBGS[key]

def _ab_via(name):
    if not name.startswith('detail'):return _wait_parent_via(name)
    def call(data):
        buf=_wait_dbg(name,data['qkv'].device)
        with _ab_ctx(name):return _one(data,dbg=buf)
    return call

def _wait_q(xs):
    import statistics
    xs=sorted(xs);assert xs
    return [round(xs[0],3),round(statistics.median(xs),3),round(xs[int(.9*(len(xs)-1))],3),round(xs[-1],3)]

def _wait_metrics(rec):
    # Low32globaltimer differences wrap modulo2^32; no cross-clock subtraction.
    def us(a,b):return (((int(b)-int(a)+(1<<31))&0xffffffff)-(1<<31))/1000
    values={k:[] for k in ('empty_wait','issue_after_empty','refetch_issue_region','consumer_ready_wait','ready_to_release','release_to_next','copy_publish_to_ready','tile_cadence')}
    coverage=[]
    for r in rec:
        assert r[4]==r[5] and r[4]>0,'producer/consumer ordinal coverage mismatch'
        for slot,ordinal in enumerate((0,16,64,128)):
            v=r[16+slot*10:26+slot*10]
            live=ordinal<r[4]
            assert (v[0]==ordinal+1)==live and (v[1]==ordinal+1)==live,'sample presence mismatch'
            if not live:continue
            coverage.append(ordinal)
            values['empty_wait'].append(us(v[2],v[3]))
            values['issue_after_empty'].append(us(v[3],v[4]))
            values['refetch_issue_region'].append(us(v[4],v[5]))
            values['consumer_ready_wait'].append(us(v[6],v[7]))
            values['ready_to_release'].append(us(v[7],v[8]))
            values['copy_publish_to_ready'].append(us(v[4],v[7]))
            if ordinal+1<r[5]:
                values['release_to_next'].append(us(v[8],v[9]))
                values['tile_cadence'].append(us(v[6],v[9]))
    assert all(values.values())
    # Report per-sample quantiles; these regions are not additive recoverable savings.
    return {'sample_counts':{str(i):coverage.count(i) for i in (0,16,64,128)},
            'region_us':{k:_wait_q(v) for k,v in values.items()},'tiles':_wait_q([r[4] for r in rec])}

def _wait_collect():
    global _WAIT_ROWS
    import time
    started=time.monotonic()
    flush=torch.empty(256<<20,device='cuda',dtype=torch.uint8)
    previous=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        fns=_ab_fns()
        for batch in (16,32):
            for length in (1,8):
                for index in range(3):
                    assert time.monotonic()-started<120,'wait collection budget'
                    seed=511019+index
                    base=_ab_shape_inputs(batch,'full',length,seed)
                    data={n:{k:v.clone() if isinstance(v,torch.Tensor) else dict(v) if isinstance(v,dict) else v for k,v in base.items()} for n in fns}
                    for k in ('qkv','k_cache','v_cache'):assert len({d[k].data_ptr() for d in data.values()})==4
                    for n,fn in fns.items():
                        out=fn(data[n]);torch.cuda.synchronize();ok,msg=_ob_check(data[n],out);assert ok,('wait math',n,msg)
                    for name in ('detailA','detailB'):
                        records=[]
                        for rep in range(2):
                            flush.zero_()
                            fns[name](data[name]);torch.cuda.synchronize()
                            with _ab_ctx(name):n=_one_plan(data[name]['config'],data[name]['page_table'].shape[1],_one_state(data[name]['qkv'].device)['resident'])[0]
                            buf=_wait_dbg(name,data[name]['qkv'].device)
                            rec=buf[1024:1024+n*56].view(n,56).cpu().long().tolist()
                            records.append(_wait_metrics(rec))
                        _WAIT_ROWS.append({'batch':batch,'length':length,'seed':seed,'arm':name,'repetitions':records})
    finally:torch.backends.cuda.matmul.allow_tf32=previous

def _wait_final():
    import json,zlib,base64
    assert len(_WAIT_ROWS)==24 and len(_AB_INMED)==12,'incomplete wait cells/inputs/timings'
    timing=[];usable=True
    for (tag,inp),m in _AB_INMED.items():
        null=max(abs(m['ctrl2']-m['ctrl']),abs(m['detailA']-m['detailB']))
        perturb=max(abs(m[d]-m[c]) for d in ('detailA','detailB') for c in ('ctrl','ctrl2'))
        limit=max(.5,.01*min(m['ctrl'],m['ctrl2']))
        passed=perturb<=limit and null<=limit;usable &= passed
        timing.append({'shape':tag,'input':inp,'us':m,'perturbation_us':perturb,'null_us':null,'limit_us':limit,'usable':passed})
    report={'schema':1,'scope':'owned full16/32 decode+verify','bf16_boundary':'common baseline parity repair; FP32 judge-style references unchanged',
        'resources':{n:{str(k):v for k,v in d.items()} for n,d in _WAIT_RES.items()},'timings':timing,'progress':_WAIT_ROWS,
        'causal_usable':usable,'classification':'sampled leader issue/wait regions, no HBM counters, no additive speedup inference'}
    raw=json.dumps(report,separators=(',',':')).encode();blob=base64.b64encode(zlib.compress(raw,9)).decode()
    assert len(blob)<=11000,'wait fullreport log budget'
    print('OWNED_WAIT_TRACE '+blob,flush=True)
