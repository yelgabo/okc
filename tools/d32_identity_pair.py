"""Build the bounded identity-rescale comparison from the existing layer harness."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import d32_attn_worker_harness as harness
import d32_attn_worker_pair as pair
import d32_identity_rescale as candidate


PROGRESS = '''
_D32_PROGRESS = {'rows': [], 'stage': 'entry'}

def _d32_emit(stage, error=None):
    import json,zlib,base64
    elapsed=time.monotonic()-_EH_START
    _D32_PROGRESS['stage']=stage
    record=dict(schema=1,pins=_EH_PINS,stage=stage,elapsed_s=elapsed,
                covered=len(_D32_PROGRESS['rows']))
    if error is not None:
        record.update(rows=_D32_PROGRESS['rows'],error=type(error).__name__+':'+str(error)[:512],
                      status='FAILED_PARTIAL_NOT_QUALIFIED')
    elif _D32_PROGRESS['rows']:
        row=_D32_PROGRESS['rows'][-1]
        record.update(last=[row['tokens'],row['context'],row['seed']],times_us=row['times_us'])
    blob=base64.b64encode(zlib.compress(json.dumps(record,separators=(',',':'),allow_nan=False).encode(),9)).decode()
    if len(blob)>11000:raise RuntimeError('D32_PARTIAL_REPORT_OVERFLOW')
    print('D32_IDENTITY_PROGRESS '+blob,flush=True)

def _d32_checkpoint(stage, deadline):
    _d32_emit(stage)
    assert time.monotonic()-_EH_START<deadline,('D32_PHASE_BUDGET',stage,deadline)
'''


def build(private_check=False):
    old, _ = harness.build(private_check)
    original = pair.build(pair.CANDIDATE.read_text())
    source, _, _, manifest = candidate.build()
    text = harness.once(old, original, source+'\n'+pair.SUPPORT)
    start = text.index('\nimport math,time\n', len(source))
    prefix, core = text[:start], text[start:]
    core = harness.once(core, '    rows=[];checks=0;fresh_checks=0;restores=0',
                        "    rows=_D32_PROGRESS['rows'];checks=0;fresh_checks=0;restores=0")
    core = harness.once(core, '    for tokens in LOADS:', '    for tokens in (32,1,8,16,128,256):')
    core = harness.once(core, '    owners=_wl_check_owners([state,*arms],readonly)[1:]',
                        "    owners=_wl_check_owners([state,*arms],readonly)[1:]\n    _d32_checkpoint('owners',210)")
    core = harness.once(core, '    transitions=_eh_transitions(arms,weights)',
                        "    transitions=_eh_transitions(arms,weights)\n    _d32_checkpoint('transitions',280)")
    core = harness.once(core, "        state=_EH_PREPARE(weights);_eh_budget()",
                        "        _D32_PROGRESS['rows']=[]\n        _d32_emit('prepare-start')\n        state=_EH_PREPARE(weights);_eh_budget()\n        _d32_checkpoint('prepared',180)")
    core = harness.once(core, '    assert time.monotonic()-_EH_START<295,\'EXPERT_PREPARE_295S\'',
                        "    assert time.monotonic()-_EH_START<280,('D32_WORK_BUDGET',_D32_PROGRESS['stage'])")
    core = harness.once(core, "                    selected=[a['_d32_probe_seen'] for a in arms]))",
                        "                    selected=[a['_d32_probe_seen'] for a in arms]))\n                _d32_checkpoint('row',280)")
    core = harness.once(core, '    except BaseException:\n        guard_mark=entry_mark;guard_ids=entry_ids\n        raise',
                        "    except BaseException as error:\n        try:_d32_emit('failure',error)\n        except BaseException:pass\n        guard_mark=entry_mark;guard_ids=entry_ids\n        raise")
    core = harness.once(core, "        _eh_budget();print('D32_ATTENTION_PAIR_REPORT '+blob,flush=True)",
                        "        _d32_checkpoint('serialize',290)\n        print('D32_ATTENTION_PAIR_REPORT '+blob,flush=True)")
    core = harness.once(core, '    except BaseException:\n        del _GUARDS[entry_mark:];_GUARDED_IDS.clear();_GUARDED_IDS.update(entry_ids)\n        raise',
                        "    except BaseException as error:\n        try:_d32_emit('serialization-failure',error)\n        except BaseException:pass\n        del _GUARDS[entry_mark:];_GUARDED_IDS.clear();_GUARDED_IDS.update(entry_ids)\n        raise")
    pins = dict(candidate=manifest['sha256'],control=candidate.scaffold.EXPECTED,
                native=manifest['native_sha256'],harness_core_sha256=hashlib.sha256(core.encode()).hexdigest(),
                progress_sha256=hashlib.sha256(PROGRESS.encode()).hexdigest())
    a=prefix.index('\n_EH_PINS='); b=prefix.index('\n',a+1)
    prefix=prefix[:a]+'\n_EH_PINS='+repr(pins)+prefix[b:]
    text=prefix+PROGRESS+core
    ast.parse(text)
    if len(text.encode())>=1_000_000:raise ValueError('private source exceeds limit')
    return text,dict(sha256=hashlib.sha256(text.encode()).hexdigest(),bytes=len(text.encode()),
                     pins=pins,offline_hold=not private_check,
                     status='CPU_BUDGET_AND_TARGET_GATES_OPEN')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--enable-private-check',action='store_true')
    args=p.parse_args()
    text,manifest=build(args.enable_private_check)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(text)
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
