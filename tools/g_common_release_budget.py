"""Exact-ID reference reuse for the frozen all12 single-arm qualifier only.

No product kernels, native source, launch, reference arithmetic or test removed.
Old 97935e7d/4e3895be artifacts remain immutable; emit a distinct output path.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import g_common_release as prior

PINS={'common':'4e3895be33d3d8f60670b4e8138eb09654ddc6d64aa930170dc89cc3f019d270',
      'short-rmw':'97935e7dc45d9fa69f5b4965bd70a5c2946cd8b829cedffb7a2aefc598cb53c7'}

SUPPORT=r'''
_GQB_CACHE=None
_GQB_START=None
_GQB_PHASE='not_started'
_GQB_CELL=None
_GQB_COUNTS={'hit':0,'miss':0,'peak_entries':0}

def _gqb_phase(phase):
    global _GQB_PHASE
    _GQB_PHASE=phase
    print('GQUAL_PHASE',phase,_GQB_CELL,round(time.monotonic()-_GQB_START,6),flush=True)

def _gqb_signature(front):
    # These are internally constructed, immutable reference banks. Version
    # checks also reject/recompute ordinary Torch mutations between checks.
    result=[]
    for name,value in front.items():
        if isinstance(value,torch.Tensor):
            result.append((name,id(value),value._version,tuple(value.shape),value.dtype,value.device))
        else:
            assert type(value) in (str,int,float,bool,type(None)),'GQUAL_REFERENCE_OWNER'
            result.append((name,value))
    return tuple(result)

def _gqb_validate(*args):
    _gqb_phase('validate_begin')
    _gq_budget(_GQB_START)
    _qual_validate(*args)
    _gq_budget(_GQB_START)
    _gqb_phase('validate_end')
'''

BACK=r'''def _gq_ref_back(front,weights,ids):
    cache=_GQB_CACHE
    signature=_gqb_signature(front) if cache is not None else None
    entry=cache.get(id(front)) if cache is not None else None
    if entry is not None:
        assert entry['front'] is front and entry['weights'] is weights,'GQUAL_REFERENCE_OWNER'
        assert entry['ids']._version==entry['ids_version'],'GQUAL_REFERENCE_IDS_MUTATED'
        assert tuple(t._version for t in entry['result'])==entry['versions'],'GQUAL_REFERENCE_RESULT_MUTATED'
    same=(entry is not None and entry['signature']==signature and
          ids.shape==entry['ids'].shape and ids.dtype==entry['ids'].dtype and
          ids.device==entry['ids'].device and torch.equal(ids,entry['ids']))
    if same:
        result=entry['result'];_GQB_COUNTS['hit']+=1
    else:
        if cache is not None:
            _gqb_phase('reference_back_miss');_gq_budget(_GQB_START)
        result=_GQ_REF_BACK(front,weights,ids)
        if cache is not None:
            _gq_budget(_GQB_START);_GQB_COUNTS['miss']+=1
            snapshot=ids.clone()
            cache[id(front)]={'front':front,'weights':weights,'signature':signature,
                'ids':snapshot,'ids_version':snapshot._version,'result':result,
                'versions':tuple(t._version for t in result)}
            assert len(cache)<=4,'GQUAL_REFERENCE_CACHE_BOUND'
            _GQB_COUNTS['peak_entries']=max(_GQB_COUNTS['peak_entries'],len(cache))
    for i,value in enumerate(result):_qual_finite(value,'g_reference_'+str(i))
    return result'''


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new,1)


def build(root,arm='common'):
    old,manifest=prior.build(root,arm)
    assert sha(old)==PINS[arm],'frozen qualifier drift'
    body=prior.BODY
    node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name=='_gq_ref_back')
    body=once(body,ast.get_source_segment(body,node),BACK)
    body=once(body,"    assert time.monotonic()-start<295,'GQUAL_PREPARE_BUDGET'",
        "    elapsed=time.monotonic()-start\n    if elapsed>=295:print('GQUAL_BUDGET_CONTEXT',_GQB_PHASE,_GQB_CELL,round(elapsed,6),flush=True)\n    assert elapsed<295,'GQUAL_PREPARE_BUDGET'")
    body=once(body,'    import copy\n    _gq_budget(start)',
        "    global _GQB_CACHE,_GQB_CELL\n    import copy\n    _gqb_phase('clone_begin');_gq_budget(start)")
    body=once(body,'    resources=_gq_resources(private);_gq_budget(start)',
        "    _gqb_phase('clone_end_resources_begin')\n    resources=_gq_resources(private);_gq_budget(start)\n    _gqb_phase('resources_end')")
    body=once(body,"            _gq_budget(start);seed=949001+tokens*1009+(layer=='full')*100",
        "            _GQB_CACHE={};_GQB_CELL=(tokens,layer)\n            _gqb_phase('cell_begin');_gq_budget(start);seed=949001+tokens*1009+(layer=='full')*100")
    body=once(body,'            base,refs=_gq_inputs(tokens,layer,seed,weights)',
        "            _gqb_phase('base_front_begin')\n            base,refs=_gq_inputs(tokens,layer,seed,weights)\n            _gqb_phase('base_front_end_fresh_front_begin')")
    body=once(body,'            fresh,fresh_refs=_gq_inputs(tokens,layer,seed+101,weights)',
        "            fresh,fresh_refs=_gq_inputs(tokens,layer,seed+101,weights)\n            _gqb_phase('fresh_front_end_input_clone_begin')")
    body=once(body,'            empty_graph,empty_out=_gq_empty(private,tokens,start);empty_checks+=2;empty_captures+=1',
        "            _gqb_phase('empty_begin')\n            empty_graph,empty_out=_gq_empty(private,tokens,start);empty_checks+=2;empty_captures+=1\n            _gqb_phase('empty_end')")
    assert body.count('_qual_validate(')==3
    body=body.replace('_qual_validate(','_gqb_validate(')
    body=once(body,'            del graph,empty_graph,output,empty_out,data,base,fresh,refs,fresh_refs,source,want,warm',
        "            _GQB_CACHE.clear();_GQB_CACHE=None\n            _gqb_phase('cell_end')\n            del graph,empty_graph,output,empty_out,data,base,fresh,refs,fresh_refs,source,want,warm")
    body=once(body,"        'native_resources':resources,'actual_attention':records}",
        "        'native_resources':resources,'actual_attention':records,'reference_cache':dict(_GQB_COUNTS)}")
    body=once(body,'    global _ma_ext,_decode_attention7,_moe',
        '    global _ma_ext,_decode_attention7,_moe,_GQB_START,_GQB_CACHE,_GQB_CELL')
    body=once(body,'    start=time.monotonic();prior_tf32=torch.backends.cuda.matmul.allow_tf32',
        "    start=time.monotonic();_GQB_START=start;_GQB_CACHE=None;_GQB_CELL=None\n    _GQB_COUNTS.update(hit=0,miss=0,peak_entries=0)\n    _gqb_phase('normal_prepare_begin')\n    prior_tf32=torch.backends.cuda.matmul.allow_tf32")
    body=once(body,'        production=_GQ_NORMAL_PREPARE(weights);_gq_budget(start)',
        "        production=_GQ_NORMAL_PREPARE(weights);_gqb_phase('normal_prepare_end');_gq_budget(start)")
    body=once(body,'    finally:\n        _ma_ext=normal_build;',
        '    finally:\n        if _GQB_CACHE is not None:_GQB_CACHE.clear()\n        _GQB_CACHE=None\n        _ma_ext=normal_build;')
    assert old.endswith(prior.BODY)
    source=old[:-len(prior.BODY)]+SUPPORT+'\n'+body
    assert len(source.encode())<1_000_000
    # Every original top-level declaration except the four prepare/reference
    # qualification functions is structurally identical, including all native
    # literals, timed entries, reference math and adaptive/cache/routing checks.
    changed={'_gq_budget','_gq_ref_back','_gq_private','prepare'}
    a=ast.parse(old);b=ast.parse(source)
    def kept(tree):
        return [ast.dump(n) for n in tree.body if not
            (isinstance(n,ast.FunctionDef) and n.name in changed) and not
            (isinstance(n,ast.FunctionDef) and n.name.startswith('_gqb_')) and not
            (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('_GQB_') for t in n.targets))]
    assert kept(a)==kept(b),'non-qualification source changed'
    return source,dict(manifest,source_sha256=sha(source),bytes=len(source.encode()),
        prior_body_sha256=manifest['body_sha256'],body_sha256=sha(body),support_sha256=sha(SUPPORT),
        prior_qualifier_sha256=PINS[arm],budget_epoch='exact-id-back-reference-cache-v1',
        cache_limit=4,reference_math_unchanged=True,all_original_validation_counts_unchanged=True,
        measured_prepare_gain=None,H200_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--arm',choices=tuple(PINS),default='common');a=p.parse_args()
    source,manifest=build(a.root,a.arm);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes(source.encode());a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
