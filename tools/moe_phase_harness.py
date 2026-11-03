"""Build sparse phase attribution with the uninstrumented control officially judged."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import expert_top100_moe_adapter as adapter
from expert_top100_moe_harness import ADAPTER_PIN, REFERENCE_PIN, NVFP4_PIN, extract
from expert_top100_pregamma import capture
from prod_size_compile_g import bindings
from cuda_issue_groups import BASE, BASE_PIN, NATIVE_PIN, ROOT, sha
from moe_phase import transform


def assignment(source, name, value):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)
    old = ast.get_source_segment(source, node)
    assert source.count(old) == 1
    return source.replace(old, name + ' = ' + repr(value))


def build(*, target_proposal=True):
    assert target_proposal is True
    prefix, ignored = adapter.build()
    assert sha(prefix) == ADAPTER_PIN
    native = bindings(BASE.read_text())['MOE_CUDA_SRC']
    assert sha(native) == NATIVE_PIN
    candidate, edits = transform(native)
    experiment = 'moe_phase'
    prefix = assignment(prefix, '_EXPERT_PAIR_EDITS', edits)
    prefix = assignment(prefix, 'EXPERT_MOE_IDENTITIES', (experiment, BASE_PIN, NATIVE_PIN, sha(candidate)))
    prefix = prefix.replace('expert_top100_pregamma', experiment)
    prefix += '\n' + SCRATCH
    reference = (ROOT / 'muse_moe_w4a16/reference.py').read_text()
    nvfp4 = (ROOT / 'muse_moe_w4a16/nvfp4.py').read_text()
    assert sha(reference) == REFERENCE_PIN and sha(nvfp4) == NVFP4_PIN
    route_source = '\n\n'.join((
        'D_HIDDEN = D_EXPERT = 4096\nN_EXPERTS = 256\nROUTED_PER_TOKEN = 8\nPOPULARITY = .03\nLOCAL = {"ep8": (0, 32), "full": (0, 256)}',
        extract(nvfp4, 'outlier_channels'), extract(reference, 'generate_input')))
    calls, routes = capture(prefix)
    alternatives = dict(bindings(prefix)['_MA_ALT_SRCS'])
    alternatives[experiment] = candidate
    candidate_prefix = prefix.replace("arm = state.get('_expert_pair_arm', False)", 'arm = True')
    candidate_calls, candidate_routes = capture(candidate_prefix + '\n_MA_ALT_SRCS = ' + repr(alternatives))
    assert candidate_routes == routes
    calls += [call for call in candidate_calls if call not in calls]
    assert len(calls) == 7
    descriptors = []
    for call in calls:
        item = dict(call)
        for key in ('cpp_sources', 'cuda_sources'):
            item[key] = sha(item[key])
        descriptors.append(item)
    contract = (ROOT / 'tools/expert_top100_moe_harness_contract.py').read_text().replace('REPEATS = 8','REPEATS = 32')
    runtime = (ROOT / 'tools/expert_top100_moe_harness_runtime.py').read_text()
    runtime = progress_runtime(runtime)
    identity = sha(prefix + route_source + contract + runtime + json.dumps(descriptors, sort_keys=True))
    source = prefix + '\n' + route_source + '\n' + contract + '\n'
    source += f'H_RUNTIME_ID = {identity!r}\nH_HOLD = False\nH_BUILD_CALLS = {descriptors!r}\n' + runtime
    assert len(source.encode()) < 1_000_000
    ast.parse(source)
    manifest = dict(experiment=experiment, source_sha256=sha(source), bytes=len(source.encode()),
                    runtime_id=identity, control_native_sha256=NATIVE_PIN, candidate_native_sha256=sha(candidate),
                    official_arm='control0', hold=False, compiler_calls=descriptors, routes=routes,
                    adapter_control_sha256=ADAPTER_PIN, contract_sha256=sha(contract), runtime_sha256=sha(runtime),
                    reference_sha256=REFERENCE_PIN, nvfp4_sha256=NVFP4_PIN,
                    public_submission=False, target_qualified=False)
    return source, manifest



def progress_runtime(runtime):
    def once(old,new):
        nonlocal runtime
        assert runtime.count(old)==1,old
        runtime=runtime.replace(old,new)
    once('def prepare(weights):','def _ph_prepare_inner(weights):')
    once('    report = None\n',
         '    report = None\n    phases=[]\n    raw_trace=None\n    stage="preparation"\n'
         '    _PH_LIVE.update(rows=rows,counters=counters,phases=phases,raw_trace=None,stage=stage,began=began,sampling=[],repeated=[])\n')
    once('                    row = _h_case(',
         '                    stage=[seed,tokens,dist]\n                    _PH_LIVE["stage"]=stage\n                    row = _h_case(')
    once('                        rows.append(row)',
         '                        rows.append(row)\n'
         '                        if tokens>=32:\n'
         '                            observed,raw=_ph_collect(states)\n'
         '                            phases.append([seed,tokens,dist,observed])\n'
         '                            if raw_trace is None:raw_trace=raw\n'
         '                            _PH_LIVE["raw_trace"]=raw_trace')
    once('            timings, ratios, nulls = paired_samples(samples)',
         '            timings, ratios, nulls = paired_samples(samples)\n'
         '            if tokens>=32:_PH_LIVE["sampling"].append([seed,tokens,dist,sampling_summary(samples)])')
    once('        return [seed, tokens, dist, timings, census, ratios, nulls] if timings is not None else None',
         '        if tokens in PH_LOADS:\n'
         '            repeated=_ph_replays(states,data,graphs,outputs,flush,deadline,counters)\n'
         '            _PH_LIVE["repeated"].append([seed,tokens,dist,repeated])\n'
         '        return [seed, tokens, dist, timings, census, ratios, nulls] if timings is not None else None')
    once("official_arm='candidate2'","official_arm='control0'")
    once('owners_bytes=owners, shared_tensor_storage=[], rows=rows, counters=counters)',
         'owners_bytes=owners, shared_tensor_storage=[], rows=rows, counters=counters,phases=phases,raw_trace=raw_trace,\n'
         '                      sampling=_PH_LIVE["sampling"],repeated=_PH_LIVE["repeated"])')
    once('    finally:\n        load_inline = original_loader\n        _h_reset(states)',
         '    except BaseException as error:\n'
         '        _PH_LIVE["primary_error"]=error\n        raise\n'
         '    finally:\n        load_inline = original_loader\n'
         '        _PH_LIVE["stage"]="cleanup"\n'
         '        try:_h_reset(states)\n'
         '        except BaseException as cleanup_error:\n'
         '            _PH_LIVE["cleanup_error"]=str(cleanup_error)[:300]\n'
         '            if "primary_error" in _PH_LIVE:raise _PH_LIVE["primary_error"] from cleanup_error\n'
         '            raise')
    once('    _expert_moe_budget(deadline)\n    report.update(',
         '    _PH_LIVE["stage"]="post-cleanup-budget"\n    _expert_moe_budget(deadline)\n    report.update(')
    once('    encoded = json.dumps(report, separators=',
         '    report["observer_valid"]=observer_decision(rows,report["sampling"])["valid"]\n'
         '    _PH_LIVE["stage"]="serialization"\n    encoded = json.dumps(report, separators=')
    once("    require(len(encoded.encode()) < CAPSULE_LIMIT, 'bounded capsule')",
         '    encoded=_ph_encode(report)\n'
         "    require(len(encoded)<12000,'bounded compressed capsule')\n"
         '    _PH_LIVE["stage"]="final-budget"')
    once("print('EXPERT_MOE_CAPSULE ' + encoded, flush=True)","print('MOE_PHASE_REPORT ' + encoded, flush=True)")
    once('    return states[2]','    return states[0]')
    once('    return _expert_moe_run(data, state, True)','    return _expert_moe_run(data, state, False)')
    analysis=(ROOT/'tools/moe_phase_analysis.py').read_text()
    return analysis+'\n'+COLLECT+'\n'+runtime+'\n'+ENTRY


SCRATCH = '''
_PH_SCRATCH_BASE=_ma_make_scratch
def _ma_make_scratch(device):
    return _PH_SCRATCH_BASE(device)+[torch.zeros(136*24,dtype=torch.int64,device=device)]
'''
COLLECT = '''
_PH_LIVE={}

def _ph_collect(states):
    observed=[];raw=None
    for state in states[2:]:
        values=state['ma_scratch'][3].cpu().tolist()
        summary,packed=phase_trace(values,state['nsm'])
        observed.append(summary)
        if raw is None:raw=packed
    return observed,raw


def _ph_replays(states,data,graphs,outputs,flush,deadline,counters):
    keys=('boundary_span_us','consumer_finish_spread_us','last_compute_to_last_exit_us')
    repeated=[[] for _ in range(2)]
    references=[_expert_moe_run(data,states[i],False).clone() for i in (0,1)]
    for repeat in range(5):
        for owner,i in enumerate((2,3)):
            _expert_moe_budget(deadline)
            flush.add_(1)
            outputs[i].fill_(17)
            graphs[i].replay()
            torch.cuda.synchronize(states[i]['device'])
            for reference in references:
                errors=_h_errors(outputs[i],reference)
                counters['worst']=[max(a,b) for a,b in zip(counters['worst'],errors)]
            _expert_moe_check_guards(states[i])
            summary,_=phase_trace(states[i]['ma_scratch'][3].cpu().tolist(),states[i]['nsm'])
            repeated[owner].append([summary[k] for k in keys])
            counters['trace_replays']=counters.get('trace_replays',0)+1
    return [[[min(x[k] for x in owner),statistics.median(x[k] for x in owner),max(x[k] for x in owner)]
             for k in range(len(keys))] for owner in repeated]


def _ph_encode(report):
    import base64,zlib
    return base64.b64encode(zlib.compress(json.dumps(report,separators=(',',':'),allow_nan=False).encode(),9)).decode()


def _ph_partial(error):
    import base64,zlib
    report={k:v for k,v in _PH_LIVE.items() if k!='primary_error'}
    report.update(runtime_id=H_RUNTIME_ID,status='FAILED_PARTIAL_NOT_QUALIFIED',
                  elapsed_s=time.monotonic()-report['began'],error=type(error).__name__+':'+str(error)[:300])
    def encode():
        return base64.b64encode(zlib.compress(json.dumps(report,separators=(',',':'),allow_nan=False).encode(),9)).decode()
    blob=encode()
    if len(blob)>=12000:
        report['raw_trace']=None
        report['raw_trace_omitted_for_failure_bound']=True
        blob=encode()
    if len(blob)>=12000:
        report['phases']=[[x[0],x[1],x[2],[[v['boundary_span_us'],v['consumer_finish_spread_us'],v['last_compute_to_last_exit_us']] for v in x[3]]] for x in report['phases']]
        report['phase_details_compacted_for_failure_bound']=True
        blob=encode()
    if len(blob)>=12000:
        report['sampling']=[];report['repeated']=[]
        report['diagnostic_details_omitted_for_failure_bound']=True
        blob=encode()
    require(len(blob)<12000,'bounded failure evidence')
    print('MOE_PHASE_PARTIAL '+blob,flush=True)
'''

ENTRY = '''
def prepare(weights):
    _PH_LIVE.clear()
    _PH_LIVE.update(rows=[],counters={},phases=[],raw_trace=None,stage='entry',began=time.monotonic())
    try:
        return _ph_prepare_inner(weights)
    except BaseException as error:
        try:_ph_partial(error)
        except BaseException as reporting_error:
            print('MOE_PHASE_REPORTING_FAILURE '+type(reporting_error).__name__+':'+str(reporting_error)[:300],flush=True)
        raise
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(source.encode())
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
