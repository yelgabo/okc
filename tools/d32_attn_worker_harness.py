"""Build a check-only matched full-layer attention comparison."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import d32_attn_worker_pair as pair
import expert_top100_harness as original
import expert_top100_pair_package as original_pair

ROOT = Path(__file__).resolve().parents[1]


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('missing or ambiguous harness anchor: ' + old[:100])
    return text.replace(old, new)


def replace_function(text, name, replacement):
    start = text.index('\ndef ' + name + '(')
    end = text.find('\ndef ', start + 1)
    if end < 0:
        raise ValueError('missing following function: ' + name)
    return text[:start] + '\n' + replacement.rstrip() + '\n' + text[end:]


def build(private_check=False):
    payload = pair.build(pair.CANDIDATE.read_text())
    previous, prior = original.build()
    previous_package, _ = original_pair.build()
    core = previous[previous.index('\nimport math,time\n', len(previous_package)):]
    old_contract = (ROOT / 'tools/expert_top100_harness_contract.py').read_text()
    new_contract = (ROOT / 'tools/d32_attn_worker_contract.py').read_text()
    core = once(core, old_contract, new_contract)
    core = replace_function(core, '_eh_native', '''def _eh_native(M=None,state=None):
    return _EH_NATIVE(M,state)''')
    core = replace_function(core, '_wl_run', '''def _wl_run(data,state):
    # The layer may catch selector errors and fall back before the probe sees them.
    if type(state.get('_d32_pair_arm')) is not bool or state.get('_d32_pair_arm') is not state.get('_d32_pair_bound'):
        raise RuntimeError('D32 arm changed after binding')
    _d32_pair_selected_module(data['config'],state)
    state['_wl_probe']=True
    state['_d32_probe_seen']=[]
    try:out=_WL_KERNEL(data,state)
    finally:state['_wl_probe']=False
    cfg=data['config']
    target=_d32_attention_scope(cfg) and state.get('d32_attention_eligible',False)
    selected=bool(state['_d32_pair_arm']) and target
    assert state['_d32_probe_seen']==[selected],('D32_ACTUAL_SELECTION',state['_d32_probe_seen'],selected)
    return out''')
    core = replace_function(core, '_eh_routes', '''def _eh_routes(state,out,start):
    ids=out[2].long().sort(-1).values
    assert bool((ids[:,1:]!=ids[:,:-1]).all()),'D32_DUPLICATE_IDS' ''')
    core = once(core, "if isinstance(value,torch.Tensor):assert bool(torch.isfinite(value).all()),'EXPERT_FRONT_NONFINITE'",
                "if isinstance(value,torch.Tensor):assert bool(torch.isfinite(value.float() if value.dtype in (torch.float8_e4m3fn,torch.float8_e5m2) else value).all()),'EXPERT_FRONT_NONFINITE'")
    core = once(core, '        _expert_pair_bind(arm,i>=2)', '        _d32_pair_bind(arm,i>=2)')
    core = once(core, "                assert all(torch.equal(outputs[0][2],out[2]) for out in outputs[1:]),'EXPERT_ROUTING_ARM_MISMATCH'",
                "                _d32_require_same_routes(outputs,arms,tokens,context,seed,'initial')")
    core = once(core, "                    assert all(torch.equal(outputs[0][2],out[2]) for out in outputs[1:]),'EXPERT_FRESH_ROUTING_MISMATCH'",
                "                    _d32_require_same_routes(outputs,arms,tokens,context,seed,'fresh-'+str(rep))")
    core = once(core, "        assert arms[0]['ma_ext'][key]._module is not arms[2]['ma_ext'][key]._module",
                "        assert arms[0]['ma_ext'][key]._module is arms[2]['ma_ext'][key]._module")
    core = once(core, '                components=[_eh_component(arm,tokens) for arm in arms]\n', '')
    core = once(core, '                samples=[[[] for _ in arms] for endpoint in range(2)]',
                '                samples=[[[] for _ in arms] for endpoint in range(1)]')
    core = once(core, '                for endpoint in range(2):', '                for endpoint in range(1):')
    core = once(core, '                            graph=captures[i][0] if endpoint==0 else components[i][0]',
                '                            graph=captures[i][0]')
    core = once(core, '                for graph,args,expected in components:_release_verify(torch,args[5],expected)\n', '')
    core = once(core, '                    component_us=[statistics.median(v) for v in samples[1]],actual_routes=[a[\'_wl_hits\'] for a in arms]))',
                "                    selected=[a['_d32_probe_seen'] for a in arms]))")
    core = once(core, '                del graph,args,expected,components,captures,outputs,data,base,fresh,refs,freshrefs,front,front8',
                '                del graph,captures,outputs,data,base,fresh,refs,freshrefs,front,front8')
    core = once(core, "        assert not state.get('_expert_pair_arm',False),'EXPERT_OFFICIAL_ARM'",
                "        assert state.get('d32_attention_eligible',False),'D32_TREATMENT_NOT_PREPARED'")
    core = once(core, "print('EXPERT_TOP100_REPORT '+blob,flush=True)",
                "print('D32_ATTENTION_PAIR_REPORT '+blob,flush=True)")
    core = once(core, "official_arm='exact607'", "official_arm='attention_treatment'")
    probe = '''
def _d32_require_same_routes(outputs,arms,tokens,context,seed,stage):
    import json
    reference=outputs[0][2]
    mismatches=[]
    for i in range(1,4):
        other=outputs[i][2]
        if not torch.equal(reference,other):
            rows=torch.nonzero(torch.any(reference!=other,dim=-1),as_tuple=False).flatten()
            sample=rows[:8].cpu().tolist()
            mismatches.append(dict(placement=i,treatment=arms[i]['_d32_pair_arm'],
                selected=arms[i].get('_d32_probe_seen'),row_count=int(rows.numel()),
                rows=sample,control_ids=reference[rows[:8]].cpu().tolist(),
                other_ids=other[rows[:8]].cpu().tolist()))
    if mismatches:
        print('D32_ROUTE_MISMATCH '+json.dumps(dict(tokens=tokens,context=context,seed=seed,
              stage=stage,mismatches=mismatches),separators=(',',':')),flush=True)
        raise AssertionError('D32_NATURAL_ROUTING_MISMATCH')

_d32_pair_selected_module = _d32_attention_module
def _d32_attention_module(cfg,state):
    module=_d32_pair_selected_module(cfg,state)
    if state.get('_wl_probe',False):
        state['_d32_probe_seen'].append(module is not None)
    return module
'''
    pins = {'candidate': pair.PIN, 'control': 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab',
            'harness_core_sha256': hashlib.sha256(core.encode()).hexdigest()}
    text = payload + probe + '\n_EH_OFFLINE_HOLD=' + repr(not private_check) + '\n_EH_PINS=' + repr(pins) + '\n' + core
    ast.parse(text)
    if len(text.encode()) >= 1_000_000:
        raise ValueError('matched source exceeds size limit')
    return text, {'sha256': hashlib.sha256(text.encode()).hexdigest(), 'bytes': len(text.encode()),
                  'pins': pins, 'offline_hold': not private_check, 'dispatch_ready': private_check,
                  'status': 'GENERATED_CPU_AND_TARGET_GATES_OPEN', 'oracle': prior['oracle']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--enable-private-check', action='store_true')
    args = parser.parse_args()
    text, manifest = build(private_check=args.enable_private_check)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(text.encode())
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
