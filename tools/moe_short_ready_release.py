"""Prepare an unqualified single-arm draft without changing the active pair."""
import argparse
import ast
import json
from pathlib import Path

from moe_control_release import build as control_release
from moe_short_ready_harness import build as paired
from moe_factor_y_harness import assignment
from moe_short_ready import transform,HELPER_PIN
from prod_size_compile_g import bindings
from expert_top100_pregamma import capture
from cuda_issue_groups import sha


def build():
    source,prior=control_release()
    assert sha(source)=='5df257afd6ae2863bc706826a9798d97b17e73830f1d10703c81bba21203eb9e'
    calls,routes=capture(source)
    helper=bindings(source)['_MA_ALT_SRCS']['moe_helper0'];assert sha(helper)==HELPER_PIN
    candidate,_=transform(helper)
    original_identity=bindings(source)['EXPERT_MOE_IDENTITIES']
    source=assignment(source,'EXPERT_MOE_IDENTITIES',('moe-short-ready-release',*original_identity[1:],sha(candidate)))
    old='    ext = state["ma_ext"].get(key)'
    assert source.count(old)==1
    source=source.replace(old,'    if key in ("tiny","tiny2"):\n        defines=tuple(defines)+("@moe_short_ready",)\n'+old)
    source+='\n_MA_ALT_SRCS["moe_short_ready"] = '+repr(candidate)+'\n'
    source=source.replace('Standalone control release candidate. Public submission requires approval.',
                          'Single-arm ready-flag release draft. Requires private qualification and public approval.')
    after,new_routes=capture(source);assert routes==new_routes
    pair,pair_manifest=paired()
    assert sha(pair)=='b6107a783c8ded6dc626f6cfa99e7e1b53201e5225bd2073f736c69125f16964'
    actual=[]
    for index,(before,call) in enumerate(zip(calls,after)):
        assert before['extra_cuda_cflags']==call['extra_cuda_cflags']
        assert (before['cuda_sources']!=call['cuda_sources'])==(index<2)
        descriptor=dict(call)
        for key in ('cpp_sources','cuda_sources'):descriptor[key]=sha(descriptor[key])
        assert descriptor in pair_manifest['compiler_calls']
        actual.append(descriptor)
    ast.parse(source)
    assert len(source.encode())<1_000_000 and '_SR_LIVE' not in source and 'EXPERT_MOE_CAPSULE' not in source
    return source,dict(source_sha256=sha(source),bytes=len(source.encode()),screen_source_sha256=pair_manifest['source_sha256'],
                       original_release_sha256=prior['source_sha256'],native_sha256=prior['native_sha256'],
                       original_helper_sha256=HELPER_PIN,short_helper_sha256=sha(candidate),compiler_calls=actual,routes=routes,
                       public_submission_approved=False,qualification='DRAFT, screen and held-out decision pending; exact-file GPU qualification required',
                       scope='Tiny/Tiny2 flags only, original MID/B4/MAIN, scale fallbacks, packing and three-buffer ABI')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();source,manifest=build();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(source);args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
