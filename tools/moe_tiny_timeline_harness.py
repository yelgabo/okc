"""Build one bounded short-load observer, with unchanged control officially judged."""
import argparse
import ast
import json
from pathlib import Path

from cuda_issue_groups import ROOT, BASE, BASE_PIN, NATIVE_PIN, sha
from prod_size_compile_g import bindings
from expert_top100_moe_adapter import build as adapter
from expert_top100_moe_harness import ADAPTER_PIN, REFERENCE_PIN, NVFP4_PIN, extract
from expert_top100_pregamma import capture
from moe_tiny_timeline import transform, HELPER_PIN
from moe_factor_y_harness import assignment


def build():
    prefix, _ = adapter()
    assert sha(prefix) == ADAPTER_PIN
    helper = bindings(BASE.read_text())['_MA_ALT_SRCS']['moe_helper0']
    revised, edits = transform(helper)
    prefix = assignment(prefix, '_EXPERT_PAIR_EDITS', [])
    prefix = assignment(prefix, 'EXPERT_MOE_IDENTITIES', ('moe_tiny_timeline', BASE_PIN, NATIVE_PIN, sha(revised)))
    prefix = prefix.replace("if arm and key in ('b4', 'main'):", "if arm and key in ('tiny', 'tiny2'):")
    prefix = prefix.replace("('@expert_top100_pregamma',)", "('@moe_tiny_timeline',)")
    prefix += '\n_MA_ALT_SRCS["moe_tiny_timeline"] = ' + repr(revised) + '\n'
    prefix += '''
_TT_SCRATCH_BASE = _ma_make_scratch
def _ma_make_scratch(device):
    return _TT_SCRATCH_BASE(device)+[torch.zeros(136*1184,dtype=torch.int64,device=device)]
'''
    reference = (ROOT / 'muse_moe_w4a16/reference.py').read_text()
    nvfp4 = (ROOT / 'muse_moe_w4a16/nvfp4.py').read_text()
    assert sha(reference) == REFERENCE_PIN and sha(nvfp4) == NVFP4_PIN
    routes = '\n\n'.join(('D_HIDDEN = D_EXPERT = 4096\nN_EXPERTS = 256\nROUTED_PER_TOKEN = 8\nPOPULARITY = .03\nLOCAL = {"ep8": (0,32), "full": (0,256)}',
                          extract(nvfp4, 'outlier_channels'), extract(reference, 'generate_input')))
    calls, dispatch = capture(prefix)
    observed, observed_dispatch = capture(prefix.replace("arm = state.get('_expert_pair_arm', False)", 'arm = True'))
    assert dispatch == observed_dispatch
    calls += [c for c in observed if c not in calls]
    assert len(calls) == 7
    descriptors = []
    for call in calls:
        item = dict(call)
        for key in ('cpp_sources', 'cuda_sources'):
            item[key] = sha(item[key])
        descriptors.append(item)
    contract = (ROOT / 'tools/expert_top100_moe_harness_contract.py').read_text()
    contract = contract.replace('BANKS = (1102031, 1102037, 1102049)', 'BANKS = (1102031,)')
    contract = contract.replace('TOKENS = (1, 4, 8, 16, 32, 33, 128, 256, 257)', 'TOKENS = LOADS')
    contract = contract.replace("DISTRIBUTIONS = ('normal', 'skewed')", "DISTRIBUTIONS = ('normal',)")
    contract = contract.replace('REPEATS = 8', 'REPEATS = 32')
    runtime = (ROOT / 'tools/expert_top100_moe_harness_runtime.py').read_text()
    runtime = runtime[:runtime.index('\ndef prepare(weights):')]
    runtime = runtime.replace('    return _H_NATIVE_LOADER(**kwargs)',
                              '    _TT_LOADED.add(H_BUILD_CALLS.index(descriptor))\n    return _H_NATIVE_LOADER(**kwargs)')
    old = '        return [seed, tokens, dist, timings, census, ratios, nulls] if timings is not None else None'
    assert runtime.count(old) == 1
    runtime = runtime.replace(old, '''        if tokens in (1,8):
            states[2]['ma_scratch'][3].zero_()
            flush.add_(1)
            outputs[2].fill_(17)
            graphs[2].replay()
            torch.cuda.synchronize(device)
            for index in (0,1):
                _h_errors(outputs[2],_expert_moe_run(data,states[index],False))
            for state in states:_expert_moe_check_guards(state)
            observed=trace(states[2]['ma_scratch'][3].cpu().tolist(),states[2]['nsm'])
            _TT_LIVE['traces'].append([tokens,observed])
        return [seed, tokens, dist, timings, census, ratios, nulls] if timings is not None else None''')
    trace_source = Path(__file__).with_name('moe_tiny_timeline.py').read_text()
    node = next(n for n in ast.parse(trace_source).body if isinstance(n, ast.FunctionDef) and n.name == 'trace')
    runtime += '\nQMAX=192\nHEADER=32\nSTRIDE=1184\n' + ast.get_source_segment(trace_source, node) + '\n' + ENTRY
    identity = sha(prefix + routes + contract + runtime + json.dumps(descriptors, sort_keys=True))
    source = prefix + '\n' + routes + '\n' + contract + '\n'
    source += f'H_RUNTIME_ID={identity!r}\nH_BUILD_CALLS={descriptors!r}\n' + runtime
    ast.parse(source)
    return source, dict(experiment='MOE-TINY-TIMELINE-01', source_sha256=sha(source), bytes=len(source.encode()),
                        runtime_id=identity, control_native_sha256=NATIVE_PIN, helper_sha256=HELPER_PIN,
                        observer_helper_sha256=sha(revised), edits=edits, compiler_calls=descriptors,
                        routes=dispatch, official_arm='control0', target_qualified=False,
                        scope='T1/T8 observer, six-load controls/correctness, one bank normal; no treatment gain')


ENTRY = '''
import base64
import zlib
_TT_LIVE={}
_TT_LOADED=set()

def _tt_costs(rows):
    costs=[]
    for row in rows:
        base=statistics.median(row[3][:2])
        obs=statistics.median(row[3][2:])
        duplicate=max(abs(row[3][0]-row[3][1]),abs(row[3][2]-row[3][3]))
        limit=max(2.,.1*base)
        noise_limit=max(.5,.02*base)
        costs.append([row[1],base,obs,obs-base,limit,duplicate,noise_limit,
                      abs(obs-base)<=limit and duplicate<=noise_limit])
    return costs

def _tt_encode(report):
    text=json.dumps(report,separators=(',',':'),allow_nan=False).encode()
    encoded=base64.b85encode(zlib.compress(text,9)).decode()
    if len(encoded)>12000:
        report['status']='failed'
        report['detail_compacted']=True
        for tokens,item in report.get('traces',[]):item.pop('critical_units',None)
        text=json.dumps(report,separators=(',',':'),allow_nan=False).encode()
        encoded=base64.b85encode(zlib.compress(text,9)).decode()
        require(len(encoded)<=12000,'bounded failure capsule')
    return encoded

def _tt_prepare_inner(weights):
    global load_inline
    began=time.monotonic()
    deadline=began+295
    states=[]
    rows=[]
    counters=dict(observations=0,restorations=0,captures=0,timed_replays=0,worst=[0.,0.])
    _TT_LIVE.update(runtime_id=H_RUNTIME_ID,status='preparing',rows=rows,counters=counters,
                    traces=[],official_arm='control0',stage='prepare',prepare_s=0.)
    original=load_inline
    primary=None
    try:
        load_inline=_h_loader
        for arm in (False,False,True,True):states.append(_expert_moe_prepare(weights,arm,deadline))
        _TT_LIVE['owners_bytes']=_h_ownership(states)
        require(all(s['nsm']==132 for s in states),'frozen H200 grid bound')
        require(all(s['ma_sc6'] and tuple(s['ma_sfx'])==('-DOKC_PRMTSC=1',) for s in states),'selected scale path')
        flush=torch.zeros(64*1024*1024,dtype=torch.int32,device=states[0]['device'])
        for tokens in TOKENS:
            _TT_LIVE['stage']=['case',tokens]
            rows.append(_h_case(states,1102031,tokens,'normal',flush,deadline,counters))
        require(len(rows)==6 and len(_TT_LIVE['traces'])==2,'complete bounded observer')
        require(sorted(_TT_LOADED)==list(range(7)),'all selected compiler requests')
        _TT_LIVE['loaded_builds']=sorted(_TT_LOADED)
    except BaseException as error:
        primary=error
        raise
    finally:
        load_inline=original
        _TT_LIVE['stage']='cleanup'
        try:_h_reset(states)
        except BaseException as error:
            _TT_LIVE['cleanup_error']=str(error)[:180]
            if primary is not None:raise primary from error
            raise
        _TT_LIVE['prepare_s']=time.monotonic()-began
    _TT_LIVE['stage']='post-cleanup-budget'
    _expert_moe_budget(deadline)
    costs=_tt_costs(rows)
    _TT_LIVE.update(status='complete',cleanup_restored=True,observer_costs=costs,
                    observer_valid=all(c[-1] for c in costs),stage='serialization')
    encoded=_tt_encode(_TT_LIVE)
    require(not _TT_LIVE.get('detail_compacted'),'success detail compaction')
    _TT_LIVE['stage']='final-budget'
    _expert_moe_budget(deadline)
    print('MOE_TINY_TIMELINE '+encoded,flush=True)
    return states[0]

def prepare(weights):
    try:return _tt_prepare_inner(weights)
    except BaseException as error:
        _TT_LIVE.update(status='failed',error=str(error)[:240])
        print('MOE_TINY_TIMELINE '+_tt_encode(_TT_LIVE),flush=True)
        raise

def kernel_fn(data,state):
    return _expert_moe_run(data,state,False)
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'edits'},indent=2))
