"""Generate one-cell routing diagnostics from the frozen matched payload."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import d32_attn_worker_harness as parent

PARENT_SHA = '9e5e5d64e89c7e6f1d98a213f6ae95233ad576cb40a49263b894432c0d696600'

SUPPORT = r'''
_D32_ROUTE_EVENTS=[]
_D32_ROUTE_DECODE=_decode_attention7
_D32_ROUTE_POOL=_pool_router_fp32
_D32_ROUTE_NATIVE=_eh_native
_D32_ROUTE_GATE=_d32_require_same_routes

def _d32_route_cpu(value):
    assert isinstance(value,torch.Tensor),'D32_OBSERVER_TENSOR_REQUIRED'
    result=value.detach().cpu().clone()
    assert bool(torch.isfinite(result.float()).all()),'D32_OBSERVER_NONFINITE'
    return result

def _d32_route_delta(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype,'D32_OBSERVER_BOUNDARY_CHANGED'
    return dict(equal=torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),
                max_abs=float((a.float()-b.float()).abs().max()) if a.numel() else 0.)

def _d32_observer_before(state,data):
    state['_d32_route_data']=data
    state['_d32_route_before']={k:_d32_route_cpu(data[k])
        for k in ('stream','attn_input','k_cache','v_cache')}

def _decode_attention7(data,state,qkv):
    out=_D32_ROUTE_DECODE(data,state,qkv)
    if state.get('_wl_probe',False):state['_d32_route_probe'].update(qkv=qkv,attention=out)
    return out

def _pool_router_fp32(h,hr,T,state,ppick):
    result=_D32_ROUTE_POOL(h,hr,T,state,ppick)
    if state.get('_wl_probe',False):
        import sys
        caller=sys._getframe(1).f_code
        value,parts=result
        probe=state['_d32_route_probe'];probe['normalized']=h
        if parts==-1:
            assert isinstance(value,tuple) and len(value)==2,'D32_OBSERVER_SPLIT_LAYOUT'
            pooled,logits=value
            assert pooled.shape==(T,4096) and logits.shape==(T,256),'D32_OBSERVER_SPLIT_SHAPE'
            probe.update(pooled=pooled,logits=logits)
            layout='split'
        else:
            assert type(parts) is int and parts>0 and value.numel()==parts*T*4608,'D32_OBSERVER_COMBINED_LAYOUT'
            assert value.dtype==torch.float32,'D32_OBSERVER_COMBINED_DTYPE'
            probe['raw_pool_router']=value
            layout='combined'
        state['_d32_route_branch']=dict(caller=caller.co_name,first_line=caller.co_firstlineno,
            pick=repr(ppick),layout=layout,parts=parts)
    return result

class _D32RouteNative:
    def __init__(self,ext,state):self.ext,self.state=ext,state
    def __getattr__(self,name):return getattr(self.ext,name)
    def okc_run(self,*args):
        self.state['_d32_route_probe'].update(local_ids=args[3],weights=args[4])
        return self.ext.okc_run(*args)

def _eh_native(M=None,state=None):
    ext=_D32_ROUTE_NATIVE(M,state)
    if state is None or not state.get('_wl_probe',False):return ext
    flags,digest=_EH_BUILDS[id(ext._module)]
    assert digest=='c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71' and '-DOKC_B4FIN=1' in flags,'D32_OBSERVER_NATIVE_SOURCE'
    state['_d32_route_native_branch']=dict(flags=list(flags),sha256=digest)
    return _D32RouteNative(ext,state)

def _d32_route_snapshot(state,ids):
    probe=state['_d32_route_probe'];branch=state['_d32_route_branch']
    assert {'qkv','attention','normalized','weights','local_ids'}<=probe.keys(),'D32_OBSERVER_MISSING_BOUNDARY'
    result={k:_d32_route_cpu(v) for k,v in probe.items()}
    if branch['layout']=='combined':
        raw=result['raw_pool_router'].view(branch['parts'],ids.shape[0],4608)
        logits=torch.zeros((ids.shape[0],256),dtype=torch.float32)
        pooled=torch.zeros((ids.shape[0],4096),dtype=torch.float32)
        for part in raw:
            logits+=part[:,4096:4352]+part[:,4352:4608]
            pooled+=part[:,:4096]
        result.update(logits=logits,pooled=pooled.to(torch.bfloat16))
    else:assert branch['layout']=='split','D32_OBSERVER_UNKNOWN_LAYOUT'
    result['ids']=_d32_route_cpu(ids)
    result.update({'before_'+k:v for k,v in state['_d32_route_before'].items()})
    result.update({'after_'+k:_d32_route_cpu(state['_d32_route_data'][k])
                   for k in ('stream','attn_input','k_cache','v_cache')})
    return result

def _d32_route_emit(reason):
    import base64,json,zlib
    report=dict(schema=1,pins=_D32_OBSERVER_PINS,reason=reason,events=_D32_ROUTE_EVENTS,
                timing_eligible=False,strict_route_gate=True)
    blob=base64.b64encode(zlib.compress(json.dumps(report,separators=(',',':'),allow_nan=False).encode(),9)).decode()
    assert len(blob)<11000,'D32_OBSERVER_LOG_BUDGET'
    print('D32_ROUTE_OBSERVER '+blob,flush=True)

def _d32_require_same_routes(outputs,arms,tokens,context,seed,stage):
    assert (tokens,context,seed)==(32,'sliding',1102031),'D32_OBSERVER_SCOPE'
    owners=[[_wl_storage(v) for v in arm['_d32_route_probe'].values()] for arm in arms]
    assert not any(_wl_overlap(owners[i],owners[j]) for i in range(4) for j in range(i)),'D32_OBSERVER_OWNER_ALIAS'
    snapshots=[_d32_route_snapshot(a,out[2]) for a,out in zip(arms,outputs)]
    comparisons=[];mismatches=[]
    for i,(arm,now) in enumerate(zip(arms,snapshots)):
        previous=arm.get('_d32_route_previous')
        assert now.keys()==snapshots[0].keys(),'D32_OBSERVER_LAYOUT_DISAGREEMENT'
        comparisons.append(dict(placement=i,selected=arm['_d32_probe_seen'],branch=arm['_d32_route_branch'],
            native=arm['_d32_route_native_branch'],
            shapes={k:[list(v.shape),str(v.dtype)] for k,v in now.items()},
            cross={k:_d32_route_delta(v,snapshots[0][k]) for k,v in now.items()},
            previous={k:_d32_route_delta(v,previous[k]) for k,v in now.items()} if previous else None))
        arm['_d32_route_previous']=now
        if not torch.equal(now['ids'],snapshots[0]['ids']):
            rows=torch.nonzero(torch.any(now['ids']!=snapshots[0]['ids'],dim=-1)).flatten()[:8]
            ids=now['ids'][rows].long();ref=snapshots[0]['ids'][rows].long()
            mismatches.append(dict(placement=i,rows=rows.tolist(),control_ids=ref.tolist(),other_ids=ids.tolist(),
                same_expert_sets=torch.equal(ids.sort(-1).values,ref.sort(-1).values),
                control_weights=snapshots[0]['weights'][rows].tolist(),other_weights=now['weights'][rows].tolist(),
                control_logits=snapshots[0]['logits'][rows].gather(1,ref).tolist(),
                other_logits=now['logits'][rows].gather(1,ids).tolist(),
                logit_provenance=arm['_d32_route_branch']['layout']))
    _D32_ROUTE_EVENTS.append(dict(tokens=tokens,context=context,seed=seed,stage=stage,
                                comparisons=comparisons,mismatches=mismatches))
    if mismatches:_d32_route_emit('ROUTING_MISMATCH')
    return _D32_ROUTE_GATE(outputs,arms,tokens,context,seed,stage)
'''


def build(private_check=False):
    source = (Path(__file__).resolve().parents[1] /
              'docs/steering/decode32/evidence/D32-ATTN-04/paired.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    source = parent.once(source, "    for tokens in LOADS:", "    for tokens in (32,):")
    source = parent.once(source, "        for context in ('sliding','full'):", "        for context in ('sliding',):")
    source = parent.once(source, "            for seed_idx,seed in enumerate(BANKS):", "            for seed_idx,seed in enumerate(BANKS[:1]):")
    source = parent.once(source, "    state['_wl_probe']=True\n    state['_d32_probe_seen']=[]",
                         "    state['_wl_probe']=True\n    state['_d32_route_probe']={}\n    state['_d32_probe_seen']=[]")
    source = parent.once(source, "                for i in order:captures[i]=_wl_capture(data[i],arms[i])",
                         "                for i in order:\n                    _d32_observer_before(arms[i],data[i])\n                    captures[i]=_wl_capture(data[i],arms[i])")
    source = parent.once(source, "                        captures[i][0].replay();torch.cuda.synchronize()",
                         "                        _d32_observer_before(arms[i],data[i])\n                        captures[i][0].replay();torch.cuda.synchronize()")
    source = parent.once(source, "                for i in range(4):_wl_copy(data[i],base);captures[i][0].replay()",
                         "                _d32_route_emit('EIGHT_FRESH_REPLAYS_COMPLETE')\n                raise RuntimeError('D32_OBSERVER_COMPLETE_NO_TIMING')")
    if not private_check:
        source = parent.once(source, '_EH_OFFLINE_HOLD=False', '_EH_OFFLINE_HOLD=True')
    pins = dict(parent=PARENT_SHA, observer=hashlib.sha256(SUPPORT.encode()).hexdigest())
    source += '\n_D32_OBSERVER_PINS=' + repr(pins) + '\n' + SUPPORT
    ast.parse(source)
    assert len(source.encode()) < 1_000_000
    return source, dict(sha256=hashlib.sha256(source.encode()).hexdigest(), bytes=len(source.encode()),
                        pins=pins, offline_hold=not private_check, timing_eligible=False,
                        scope='T32 sliding seed1102031, one initial and eight fresh replays, four independent owners')


def read_report(log, pins):
    import base64
    import math
    import zlib
    lines = [l.split(' ', 1)[1] for l in log.splitlines() if l.startswith('D32_ROUTE_OBSERVER ')]
    if len(lines) != 1:
        raise ValueError('expected one observer report')
    report = json.loads(zlib.decompress(base64.b64decode(lines[0], validate=True)))
    if (report.get('pins') != pins or report.get('timing_eligible') is not False
            or report.get('strict_route_gate') is not True):
        raise ValueError('observer identity or gate mismatch')
    events = report['events']
    if not 1 <= len(events) <= 9:
        raise ValueError('invalid observer event coverage')
    if any(e['mismatches'] for e in events[:-1]):
        raise ValueError('continued after strict route failure')
    if report['reason'] == 'EIGHT_FRESH_REPLAYS_COMPLETE':
        if len(events) != 9 or any(e['mismatches'] for e in events):
            raise ValueError('incomplete stable diagnostic')
    elif report['reason'] == 'ROUTING_MISMATCH':
        if not events[-1]['mismatches'] or 'D32_ROUTE_MISMATCH ' not in log:
            raise ValueError('missing strict routing failure')
    else:
        raise ValueError('unknown diagnostic reason')
    for n, event in enumerate(events):
        if ((event['tokens'], event['context'], event['seed']) != (32, 'sliding', 1102031)
                or event['stage'] != ('initial' if n == 0 else 'fresh-'+str(n-1))
                or len(event['comparisons']) != 4):
            raise ValueError('observer scope or replay order mismatch')
        for i, comparison in enumerate(event['comparisons']):
            if (comparison['placement'] != i or comparison['selected'] != [False]
                    or comparison['branch']['layout'] not in ('split', 'combined')
                    or '-DOKC_B4FIN=1' not in comparison['native']['flags']
                    or comparison['native']['sha256'] != 'c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71'):
                raise ValueError('observer selection mismatch')
            required = {'qkv','attention','normalized','pooled','logits','weights','local_ids','ids'}
            required |= {when+'_'+k for when in ('before','after')
                         for k in ('stream','attn_input','k_cache','v_cache')}
            if (not required <= comparison['cross'].keys()
                    or (comparison['previous'] is None) != (n == 0)):
                raise ValueError('missing boundary or previous replay coverage')
            for kind in ('cross', 'previous'):
                for metric in (comparison[kind] or {}).values():
                    if (type(metric['equal']) is not bool or not math.isfinite(metric['max_abs'])
                            or metric['max_abs'] < 0):
                        raise ValueError('invalid boundary comparison')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--enable-private-check', action='store_true')
    args = parser.parse_args()
    source, manifest = build(args.enable_private_check)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))
