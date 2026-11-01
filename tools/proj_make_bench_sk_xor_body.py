"""Emitted private diagnostic; normal C2 kernel_fn remains unchanged."""
BODY = r''' 
# Private SK XOR experiment. No runtime file reads, subprocesses or reference imports.
import base64 as _sx64
import zlib as _sxzip
import json as _sxjson
import math as _sxmath
import random as _sxrandom
import hashlib as _sxhash
import struct as _sxstruct

_SX_PIN = @@PIN@@
_SX_SEEDS = (903151, 903157, 903163)
_SX_ROUNDS = 24
_SX_TARGET = {('qkv_gate',8):'qkv8F', ('qkv_gate',16):'qkv16F',
 ('qkv_gate',32):'qkv32', ('o',16):'o16F', ('o',32):'o32F',
 ('pooler',16):'pool16F', ('pooler',32):'pool32F',
 ('unpooler',8):'slice8F', ('unpooler',16):'unp16F', ('unpooler',32):'unp32F'}
_SX_LAYERS = ('qkv_gate','o','pooler','unpooler','router')
_SX_MS = (1,8,16,32,128,256)
_SX_ARMS = ('c0','c1','t0','t1','anchor')
_SX_NORMAL_PREPARE = prepare
_SX_NORMAL_KERNEL = kernel_fn

def _sx_guard(start):
    assert time.monotonic()-start < 295, 'SX incomplete: prepare deadline295s'

def _sx_storage(t):
    assert isinstance(t,torch.Tensor)
    if t.numel()==0:return None
    # Full backing-storage interval rejects aliases/views, conservatively.
    s=t.untyped_storage()
    return (str(t.device),s.data_ptr(),s.data_ptr()+s.nbytes())

def _sx_disjoint(groups):
    intervals=[]
    for owner,ts in groups.items():
        for t in ts:
            x=_sx_storage(t)
            if x is None:continue
            for old_owner,old in intervals:
                assert owner==old_owner or x[0]!=old[0] or x[2]<=old[1] or old[2]<=x[1], ('SX mutable alias',owner,old_owner)
            intervals.append((owner,x))

def _sx_reference(x,w,n,router=False):
    # Official reference output boundary: BF16 projections, FP32 router.
    return (x.double()@w[:n].double().T).to(torch.float32 if router else torch.bfloat16)

def _sx_check(out,ref,x,router=False):
    assert out.shape==ref.shape and out.dtype==ref.dtype and torch.isfinite(out).all(), 'SX shape/dtype/finite'
    d=(out.double()-ref.double());r=ref.double()
    tol=1e-4 if router else 1e-2
    rel=(d.norm()/r.norm().clamp_min(1e-30)).item()
    row=(d.norm(dim=1)/r.norm(dim=1).clamp_min(1e-30)).max().item()
    assert rel<=tol and row<=5*tol, ('SX official math',rel,row,tol)
    probes=(x!=0).sum(1)==1
    if probes.any():
        p=(d[probes].norm(dim=1)/r[probes].norm(dim=1).clamp_min(1e-30)).max().item()
        assert p<=1e-3, ('SX onehot',p)
    return [rel,row]

def _sx_x(m,k,gen,dev,onehot=False,outlier=False):
    x=torch.randn((m,k),generator=gen,device=dev)
    if outlier:x[:,torch.randperm(k,generator=gen,device=dev)[:max(1,k//100)]]*=30
    if onehot:
        x.zero_()
        cols=(0,15,16,63,64,k-1)
        for i in range(m):x[i,cols[i%len(cols)]]=(-1 if i%2 else 1)*2**(3+i%5)
    else:
        # Fixed probes reproduce the official need for weight precision.
        for i in range(0,m,8):
            x[i].zero_();x[i,(17+997*i)%k]=2**(3+i%5)
    return x.to(torch.bfloat16)

def _sx_anchor_id(fn):
    vals=dict(zip(fn.__code__.co_freevars,(c.cell_contents for c in (fn.__closure__ or ()))))
    return {'code':fn.__code__.co_name, 'cid':vals.get('cid'), 'full':vals.get('full'),
            'vars':list(fn.__code__.co_freevars)}

def _sx_validate_anchor(fn,name,m):
    d=_sx_anchor_id(fn)
    # Actual prepared pick must be this native SK closure, not a fallback or another tuner winner.
    assert set(('cid','full','stream','tab','pmax','diag')).issubset(d['vars']), ('SX native C2pick missing',name,d)
    assert fn.__code__ is next(c for c in _sk_runner.__code__.co_consts if getattr(c,'co_name',None)=='run'), 'SX unexpected anchor implementation'
    vals=dict(zip(fn.__code__.co_freevars,(c.cell_contents for c in fn.__closure__)))
    actual=vals['full'] if vals['full'] is not None and m==SK_CONFIGS[vals['cid']][2] else vals['cid']
    assert actual==SK_INDEX[name], ('SX C2pick changed',name,actual,d)
    return d

def _sx_layout(name,variant,pmax):
    c=SK_CONFIGS[SK_INDEX[name]]
    _,mt,tokr,nt,nwr,wsk,upw,nstage,comp,hl,nacc,cl,_,fuse=c
    assert comp and not hl and cl==1 and not fuse and not (nacc&16)
    nw=nwr*wsk;r=nwr*nt*8;ups=wsk*upw
    payload=nw*upw*nt*768
    headers=((nw*upw*nt*4+15)&~15) if variant else 0
    xby=ups*tokr*128;sst=(xby+payload+headers+1023)&~1023
    nesc=0 if variant else min(4,nstage)
    csz=(16*mt*(r+1)+3)&~3
    fixed=1024+nstage*sst+(16*nstage+32)+4*nesc*csz+128+16
    dynamic=fixed+(0 if variant else (2*pmax+r)*4)
    return {'static':0,'dynamic':dynamic,'threads':32*(nw+1+nesc),
            'headers_per_stage':headers,'stage_stride':sst,'fixed':fixed,'escape_warps':nesc}

def _sx_capture(fn,x):
    stream=torch.cuda.Stream(device=x.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(x);fn(x)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):out=fn(x)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    g.replay();torch.cuda.synchronize()
    return g,out,stream

def _sx_cold(graphs,flush,rng,start):
    times={k:[] for k in graphs}
    for rd in range(_SX_ROUNDS):
        _sx_guard(start);order=list(graphs);rng.shuffle(order);events=[]
        for tag in order:
            flush.zero_();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();graphs[tag].replay();b.record();events.append((tag,a,b))
        torch.cuda.synchronize()
        for tag,a,b in events:times[tag].append(1000*a.elapsed_time(b))
    return times

@@MODEL@@

@@CODEC@@

def _sx_bench(weights,state,start):
    _sx_guard(start);dev=weights['router'].device
    assert set(weights)==set(_SX_LAYERS), 'SX allfive actual supplied weights required'
    modules=[xor_build('sx_a'),xor_build('sx_b')]
    assert modules[0] is not modules[1]
    _sx_guard(start)
    runners={};infos={};anchors={};all_diags={};checks=0;captures=0;replay_checks=0;post_checks=0;onehot_checks=0;outlier_checks=0
    for (layer,m),name in _SX_TARGET.items():
        w=weights[layer]; packs=[xor_pack(name,w,v) for v in (0,1)]
        assert all(p is not None for p in packs), ('SX pack rejected',name)
        pair={};detail={}
        for i,tag in enumerate(_SX_ARMS[:4]):
            v=int(tag.startswith('t'));fn,info=xor_runner(modules[i%2],name,w,v,packed=packs[v])
            pair[tag]=fn;detail[tag]=info
        anchor=state['pick'].get((layer,m));assert anchor is not None
        anchors[(layer,m)]=_sx_validate_anchor(anchor,name,m);pair['anchor']=anchor
        closure=dict(zip(anchor.__code__.co_freevars,(c.cell_contents for c in anchor.__closure__)))
        all_diags.setdefault('C2_anchor',[]).append(closure['diag'])
        all_diags.update({name+':'+a:[detail[a]['private_diag']] for a in detail})
        _sx_disjoint(all_diags)
        # Occupancy/launch geometry must match; register change explicitly recorded.
        rv=[list(detail[a]['resources']) for a in detail]
        layouts=[_sx_layout(name,int(a.startswith('t')),detail[a]['pack']['pmax']) for a in detail]
        assert all(len(v)==7 and v[1]==0 and v[6]>0 for v in rv), ('SX resources',name,rv)
        for v,layout in zip(rv,layouts):
            assert v[2]==layout['static'] and v[3]==layout['dynamic'] and v[4]==v[5]==layout['threads'], ('SX actual/analytical layout mismatch',name,v,layout)
        assert rv[0]==rv[1] and rv[2]==rv[3], ('SX build resource mismatch',name,rv)
        assert all(v[6]>=rv[0][6] for v in rv[2:]), ('SX resident admission regressed',name,rv)
        for a,v,layout in zip(detail,rv,layouts):
            detail[a]['expected_layout']=layout
            detail[a]['register_thread_product']=v[0]*v[4]
        runners[(layer,m)]=pair;infos[name]={a:{**{k:v for k,v in detail[a].items() if k not in ('immutable','pack','private_diag')},'pack_metadata':{k:v for k,v in detail[a]['pack'].items() if isinstance(v,(int,str,float,bool))},'immutable_bytes':[t.untyped_storage().nbytes() for t in detail[a]['immutable']]} for a in detail}
        # Supplemental onehot and nearbyM test uses explicit non-FULL native instantiations.
        gen=torch.Generator(device=dev).manual_seed(6021+m)
        for mm in (m,max(1,m-1)):
            for bank in ('onehot','outlier'):
                x=_sx_x(mm,w.shape[1],gen,dev,onehot=bank=='onehot',outlier=bank=='outlier');n=1024 if name=='slice8F' else w.shape[0]
                ref=_sx_reference(x,w,n)
                for i,tag in enumerate(_SX_ARMS[:4]):
                    if mm==m:fn=pair[tag]
                    else:
                        fn,near_info=xor_runner(modules[i%2],name,w,int(tag.startswith('t')),nearby=True,packed=packs[int(tag.startswith('t'))])
                        all_diags[name+':nearby:'+tag+':'+bank]=[near_info['private_diag']]
                        _sx_disjoint(all_diags)
                    _sx_check(fn(x),ref,x)
                    if bank=='onehot':onehot_checks+=1
                    else:outlier_checks+=1
        _sx_guard(start)
    flush=torch.empty(256<<20,dtype=torch.uint8,device=dev)
    report={'format':'PROJ_SK_XOR_V1','pins':_SX_PIN,'rounds':_SX_ROUNDS,'seeds':list(_SX_SEEDS),'rows':[],'resources':infos,
            'anchors':{str(k):v for k,v in anchors.items()},'weight_inventory':{k:{'shape':list(w.shape),'dtype':str(w.dtype),'bytes':w.numel()*w.element_size()} for k,w in weights.items()},
            'build_names':[m.__name__ for m in modules],
            'common_scope':'Other20 cells actual C2 single common arm; no paired/null inference there.'}
    for seed in _SX_SEEDS:
        rows=[];gen=torch.Generator(device=dev).manual_seed(seed)
        for m in _SX_MS:
            for layer in _SX_LAYERS:
                _sx_guard(start);w=weights[layer];k=w.shape[1];n=1024 if layer=='unpooler' and m<=8 else w.shape[0]
                target=(layer,m) in runners
                fns=runners[(layer,m)] if target else {'common':lambda x,layer=layer,m=m,n=n,k=k:_SX_NORMAL_KERNEL({'x':x,'config':{'layer':layer,'m':m,'n':n,'k':k}},state)}
                fresh=_sx_x(m,k,gen,dev);ref=_sx_reference(fresh,w,n,layer=='router')
                xs={a:fresh.clone() for a in fns};caps={a:_sx_capture(fn,xs[a]) for a,fn in fns.items()}
                _sx_disjoint({a:[xs[a],caps[a][1]]+([] if a=='anchor' or not target else all_diags[_SX_TARGET[layer,m]+':'+a]) for a in caps});captures+=len(caps)
                for a in caps:_sx_check(caps[a][1],ref,xs[a],layer=='router');checks+=1
                for replay in range(2):
                    _sx_guard(start);fresh=_sx_x(m,k,gen,dev);ref=_sx_reference(fresh,w,n,layer=='router')
                    for a in caps:xs[a].copy_(fresh);caps[a][0].replay()
                    torch.cuda.synchronize()
                    for a in caps:_sx_check(caps[a][1],ref,xs[a],layer=='router');checks+=1;replay_checks+=1
                raw=_sx_cold({a:caps[a][0] for a in caps},flush,_sxrandom.Random(seed+m+len(layer)),start)
                for a in caps:_sx_check(caps[a][1],ref,xs[a],layer=='router');checks+=1;post_checks+=1
                rows.append({'layer':layer,'m':m,'seed':seed,'target':target,'samples_us':raw})
                del caps,xs,ref
        assert len(rows)==30 and sum(r['target'] for r in rows)==10
        report['rows'].extend(rows)
    _sx_guard(start)
    for info in infos.values():
        assert len(info)==4
    assert (captures,checks,replay_checks,post_checks,onehot_checks,outlier_checks)==(210,840,420,210,80,80), ('SX incomplete checks',captures,checks,replay_checks,post_checks,onehot_checks,outlier_checks)
    for tensors in all_diags.values():
        for diag in tensors:assert torch.count_nonzero(diag).item()==0, 'SX device diagnostics'
    report['checks']={'captured_placements':captures,'oracle_checks':checks,'fresh_replay_checks':replay_checks,'post_timing_checks':post_checks,'onehot_nearby_checks':onehot_checks,'outlier_nearby_checks':outlier_checks}
    report['elapsed']=time.monotonic()-start;report['complete']=len(report['rows'])==90
    report['raw_gate']=_sx_score_model(report['rows']);report['raw_pass']=report['raw_gate']['keep']
    print('SXPINS '+_sxjson.dumps({'pins':_SX_PIN,'checks':report['checks'],'rows':len(report['rows']),'rounds':_SX_ROUNDS},separators=(',',':')),flush=True)
    print('SXREPORT1 '+_sx_encode(report),flush=True)
    print('SXDONE coverage=3x30 target=10 common=20 raw_pass='+str(report['raw_pass'])+' offsets=future_same_receipt_reader_only',flush=True)
    # Losing performance is still a complete experiment; incomplete/math/resource gates throw.

def prepare(weights):
    start=time.monotonic()
    state=_SX_NORMAL_PREPARE(weights)
    _sx_guard(start);_sx_bench(weights,state,start);_sx_guard(start)
    return state
'''
