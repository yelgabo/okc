"""Prepare fixed held-out banks only if the active short-ready screen passes."""
import argparse
import ast
import json
from pathlib import Path
import types

import moe_short_ready_harness as screen
import moe_short_ready_read as owning
from moe_short_gate import short_decision
from moe_factor_y_harness import assignment
from cuda_issue_groups import sha

BANKS=(2203011,2203017,2203029)
SCREEN='b6107a783c8ded6dc626f6cfa99e7e1b53201e5225bd2073f736c69125f16964'
_gate=types.FunctionType(short_decision.__code__,dict(short_decision.__globals__,BANKS=BANKS))


def build(*,target_proposal=True):
    assert target_proposal is True
    source,prior=screen.build()
    assert sha(source)==SCREEN
    start=source.index('"""Frozen standalone gates, embedded verbatim in the payload."""')
    end=source.index('\nH_RUNTIME_ID=')
    assert sha(source[start:end])==prior['contract_sha256']
    source=assignment(source,'BANKS',BANKS)
    runtime_id=sha('MOE-SHORT-READY-CONFIRM-01\n'+source)
    source=assignment(source,'H_RUNTIME_ID',runtime_id)
    start=source.index('"""Frozen standalone gates, embedded verbatim in the payload."""')
    end=source.index('\nH_RUNTIME_ID =')
    manifest=dict(prior,experiment='MOE-SHORT-READY-CONFIRM-01',source_sha256=sha(source),bytes=len(source.encode()),
                  runtime_id=runtime_id,contract_sha256=sha(source[start:end]),screen_source_sha256=SCREEN,
                  screen_job='e3e2495464f34b2c',heldout_banks=list(BANKS),dispatch_condition='complete frozen screen pass only')
    ast.parse(source)
    return source,manifest


def read(folder,job):
    prior_build,prior_gate=owning.build,owning.short_decision
    try:
        owning.build=build;owning.short_decision=_gate
        return owning.read(folder,job)
    finally:owning.build=prior_build;owning.short_decision=prior_gate


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);p.add_argument('--folder',type=Path);p.add_argument('--job')
    args=p.parse_args()
    if args.out:
        source,manifest=build();args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(source);args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({k:v for k,v in manifest.items() if k!='compiler_calls'},indent=2))
    elif args.folder and args.job:
        result=read(args.folder,args.job);print(json.dumps({k:v for k,v in result.items() if k!='report'},indent=2))
    else:p.error('provide --out or --folder and --job')
