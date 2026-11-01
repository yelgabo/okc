"""Prepare a review candidate from the user's approved upstream layer pin."""
import argparse
import hashlib
from pathlib import Path
import re
from dl_fork_attention import patch_attention
from dl_fork_projection import patch_projection

SOURCE_SHA = '82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19'
UPSTREAM = '8e30e1be994db4310794b8d04e90d18d895deb84'


def replace(source, before, after, count=1):
    if source.count(before) != count:
        raise ValueError(f'unexpected source count for {before[:60]}')
    return source.replace(before, after)


def build(raw, notices):
    raw = raw.replace(b'\r\n',b'\n')
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('unrecognized upstream layer')
    source = raw.decode('utf-8').replace('\r\n','\n')
    header, body = source.split('\n', 1)
    source = (header + '\n# Modified by yelgabo, 2026-11-01. Architecture review fork.\n'
        + f'# Based on keell0renz/muse-open-kernel-challenge at {UPSTREAM} (Apache-2.0).\n'
        + '# Retains the upstream credits below. Full upstream LICENSE and NOTICE follow the code.\n'
        + body)
    # The deferred two-row GEMM changes the location of the BF16 activation round.
    # Its carry programs also load producer partials before a PDL wait. Select the
    # existing small-token path which materializes the BF16 pre-FFN activation.
    source = replace(source, 'GLE_T = {1: True}', 'GLE_T = {}')
    source = replace(source, 'FORCE_GLUEV2 = None', 'FORCE_GLUEV2 = False')
    # All three route variants must see the BF16 pooler output for pre-expert RMS.
    bf16 = '.to(tl.bfloat16).to(tl.float32)'
    for line, count in [('pooled = tl.load(ALL + t * 4608 + c)', 2),
                        ('pooled=tl.load(ALL+t*4608+c)', 1),
                        ('vals=tl.load(ALL+t*4608+oc)', 1)]:
        source = replace(source, line+'\n', line+bf16+'\n', count)
    for line, following in [('        pooled += tl.load(b + c)', '    sc = '),
                            ('        pooled += tl.load(P + (s * T + t) * 4608 + c)', '    tl.store(POOL')]:
        source = replace(source, line+'\n'+following,
                         line+'\n    pooled = pooled'+bf16+'\n'+following)
    # The alternative boundary implementations receive FP32 split-K partials
    # for some fallbacks. Round the reconstructed projection before its norm.
    def edit_function(text, name, edits):
        start = text.index('def '+name+'(')
        stop = text.index('\n\n', text.index('\n    ', start))
        block = text[start:stop]
        for before, after in edits:
            block = replace(block,before,after)
        return text[:start]+block+text[stop:]

    source = edit_function(source,'_boundary1',[
        ('    n = a * tl.rsqrt', '    a = a'+bf16+'\n    n = a * tl.rsqrt'),
        ('b1v * n\n','b1v * n'+bf16+'\n')])
    source = edit_function(source,'_boundary2',[
        ('    s1v = tl.load(S1 + t * 8192 + offs)\n', ''),
        ('    f = tl.zeros([8192]', '    s1v = tl.load(S1 + t * 8192 + offs)\n    f = tl.zeros([8192]'),
        ('    f = f * gp', '    f = (f'+bf16+' * gp)'+bf16),
        ('b2v * n\n','b2v * n'+bf16+'\n')])
    for name in ('_graft_row_boundary','_graft_row_boundary_pre'):
        source = edit_function(source,name,[(
            'c < F, 0).to(_gl.float32)',
            'c < F, 0).to(_gl.bfloat16).to(_gl.float32)')])
    # Ordinary weights may be prefetched before the wait. Move the carry/tail data
    # reads after it so visibility does not depend on transitivity across the chain.
    start = source.index('def _sl_cluster_finish(')
    stop = source.index('\nSL_CLUSTER_CONFIG=', start)
    original = source[start:stop]
    wait = '    if USE_PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n'
    body = replace(original, wait, '')
    body = replace(body, '    carry = tl.load(', wait+'    carry = tl.load(')
    source = replace(source, original, body)
    for kernel, count in [('void glue_cluster_boundary(GlueArgs a) {',2),
                          ('BF* out, float* next, int rows) {\n  constexpr int J=1024/P',1)]:
        head,sep,rest=kernel.partition('{')
        source=replace(source,kernel,head+sep+'\n  asm volatile("griddepcontrol.wait;" ::: "memory");'+rest,count)
    # The upstream preparation benchmark compares captured outputs before replay.
    # Omit that diagnostic and its extra extension builds from the candidate.
    source = replace(source, 'MFAB_LAYER = True', 'MFAB_LAYER = False')
    source = patch_attention(source)
    # Pass an immutable weight-prefetch descriptor by value. A host launch on
    # another thread or state must never replace the pointer for this launch.
    source = replace(source, 'pdl=None, attention_state=None, **override):',
                     'pdl=None, attention_state=None, prefetch=(0,0), **override):')
    source = replace(source, '*(_PF7 or (0, 0))', '*prefetch')
    source = replace(source, '    global _PF7\n', '')
    source = replace(source, '    _PF7 = None\n', '    prefetch = (0,0)\n')
    source = replace(source, '        _PF7 = (_ps.data_ptr()', '        prefetch = (_ps.data_ptr()')
    source = replace(source, 'attention_state=state, **pdl)', 'attention_state=state, prefetch=prefetch, **pdl)')
    source = replace(source,'    _PF7 = None   # the pointer belongs to this call\'s state only\n','')
    # The normal short-boundary b1 already computes and stores the router copy.
    # Its GLE extension uses different normalization arithmetic and a global
    # weight pointer to form a second copy. Use the materialized branch.
    source = replace(source,'GLE_B1X = True','GLE_B1X = False')
    source = patch_projection(source)
    # Establish a sequential reference for G1. Some graft calls set the
    # launch attribute directly, bypassing the top-level PDL switch.
    source = replace(source,'\nPDL = True\n','\nPDL = False\n')
    source = replace(source,'launch_pdl=True','launch_pdl=False',14)
    source, count = re.subn(r'(programmaticStreamSerializationAllowed\s*=\s*)1;',r'\g<1>0;',source)
    assert count == 7
    source += '\n'+'\n'.join('# '+line for line in notices.splitlines())+'\n'
    compile(source,'<attributed layer>','exec')
    return source.encode('utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    repo = args.upstream.parent.parent
    notices = '\n'.join((repo/name).read_text(encoding='utf-8') for name in ('LICENSE','NOTICE'))
    payload=build(args.upstream.read_bytes(), notices)
    if len(payload)>=1_000_000:
        raise ValueError('payload exceeds judge limit')
    args.output.write_bytes(payload)
    print(len(payload), hashlib.sha256(payload).hexdigest())
