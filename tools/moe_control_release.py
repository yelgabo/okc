"""Prepare the current MoE control as a standalone release candidate."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from expert_top100_moe_adapter import build as adapter,CONTROL_NATIVE
from expert_top100_pregamma import capture
from prod_size_compile_g import bindings


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def build():
    source,prior=adapter()
    tree=ast.parse(source);lines=source.splitlines(keepends=True)
    removed={'_expert_pair_source','expert_moe_prepare_candidate','expert_moe_prepare_four','expert_moe_candidate_fn'}
    for node in reversed(tree.body):
        discard=isinstance(node,ast.FunctionDef) and node.name in removed
        discard|=isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('_EXPERT_PAIR_EDITS','EXPERT_MOE_DISPATCH_READY')
        discard|=isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Subscript) and ast.get_source_segment(source,node.targets[0])=="_MA_ALT_SRCS['expert_top100_pregamma']"
        if discard:lines[node.lineno-1:node.end_lineno]=[]
    source=''.join(lines)
    identity=next(node for node in ast.parse(source).body if isinstance(node,ast.Assign)
                  and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='EXPERT_MOE_IDENTITIES')
    source=source.replace(ast.get_source_segment(source,identity),
                          'EXPERT_MOE_IDENTITIES = '+repr(('moe-control-release',prior['base_sha256'],CONTROL_NATIVE,prior['helper_sha256'])))
    source=source.replace('PRIVATE SOURCE-ONLY ADAPTER. dispatch_ready=False; never submit publicly.',
                          'Standalone control release candidate. Public submission requires approval.')
    old="    if arm and key in ('b4', 'main'):\n        defines = tuple(defines) + ('@expert_top100_pregamma',)\n"
    assert source.count(old)==1
    source=source.replace(old,'')
    old="if type(arm) is not bool or '_expert_pair_arm_bound' in state:"
    assert source.count(old)==1
    source=source.replace(old,"if arm is not False or '_expert_pair_arm_bound' in state:")
    source=source.replace('_expert_pair_bind','_moe_bind_control')
    old="        if not state['ma_sfx']:\n            raise RuntimeError('EXPERT_MOE_NATIVE_SCALE_CONTRACT_NO_FALLBACK')\n"
    assert source.count(old)==1
    source=source.replace(old,'')
    old="        if set(state['ma_ext']) != {'tiny', 'tiny2', 'mid', 'b4', 'main'}:"
    assert source.count(old)==1
    source=source.replace(old,"        expected={'tiny','tiny2','mid','main'} | ({'b4'} if state['ma_sfx'] else set())\n        if set(state['ma_ext']) != expected:")
    assert 'expert_top100_pregamma' not in source and '_EXPERT_PAIR_EDITS' not in source
    assert bindings(source)['MOE_CUDA_SRC']==bindings(adapter()[0])['MOE_CUDA_SRC']
    calls,routes=capture(source);old_calls,old_routes=capture(adapter()[0])
    assert calls==old_calls and routes==old_routes and len(calls)==5
    ast.parse(source)
    descriptors=[]
    for call in calls:
        call=dict(call)
        for key in ('cuda_sources','cpp_sources'):call[key]=sha(call[key])
        descriptors.append(call)
    return source,dict(source_sha256=sha(source),bytes=len(source.encode()),native_sha256=CONTROL_NATIVE,
                       helper_sha256=prior['helper_sha256'],adapter_control_sha256=prior['source_sha256'],
                       compiler_calls=descriptors,routes=routes,judged_arm='control',public_submission_approved=False,
                       qualification='pending exact-file private check',
                       scope='Current c7 B4/MAIN, original tiny/tiny2/MID and scale fallbacks, original three scratch buffers')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();source,manifest=build();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(source);args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
