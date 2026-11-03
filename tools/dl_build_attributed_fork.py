"""Prepare a review candidate from the user's approved upstream layer pin."""
import argparse
import ast
import hashlib
from pathlib import Path
import re
from dl_fork_attention import patch_attention
from dl_fork_projection import patch_projection
from dl_fork_owned_moe import module_source, patch_owned_moe, strip_python_comments

SOURCE_SHA = '82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19'
UPSTREAM = '8e30e1be994db4310794b8d04e90d18d895deb84'


def replace(source, before, after, count=1):
    if source.count(before) != count:
        raise ValueError(f'unexpected source count for {before[:60]}')
    return source.replace(before, after)


def verify_gle_generator(source):
    """Execute the actual generated payload's fused pooler/router source builder on CPU."""
    functions = {'_gle_source', 'bfp__sk2_geom'}
    constants = {'bfp_SK2_CONFIGS', 'bfp_SK2_PICK', 'bfp_SK2_K', 'bfp_SK2_SRC',
                 'bfp_CUDA_SRC', 'bfp_MBAR_HELPERS', 'bfp_MAP_SRC'}
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(nodes, []), '<gle generators>', 'exec'), namespace)
    cuda = namespace['_gle_source']()
    assert 'static thread_local GleEpi g_gle;' in cuda and 'void gle_pr(' in cuda
    assert '@@' not in cuda


def build(raw, notices, owned, experts="owned"):
    if experts not in ("owned", "native"):
        raise ValueError("unknown expert backend")
    raw = raw.replace(b'\r\n',b'\n')
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('unrecognized upstream layer')
    source = raw.decode('utf-8').replace('\r\n','\n')
    if experts == 'native':
        from dl_fork_moe import HOST_BRIDGE, patch_moe
        source = patch_moe(source)
        if not source.endswith(HOST_BRIDGE):
            raise ValueError('native host bridge missing')
        source = source[:-len(HOST_BRIDGE)]
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
    if experts == 'owned':
        source = replace(source, 'MFAB_LAYER = True', 'MFAB_LAYER = False')
    assert 'MFAB_LAYER = False' in source
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
    # Each host launch copies its own argument into the kernel Params.
    source = replace(source,'static GleEpi g_gle;', 'static thread_local GleEpi g_gle;')
    source = replace(source,'    _gle_ext()\n    return _gle_prior_prepare(weights)',
                     '    return _gle_prior_prepare(weights)')
    source = replace(source,'    _gle_b1x_module()\n    return _gle_b1x_prior_prepare(weights)',
                     '    return _gle_b1x_prior_prepare(weights)')
    source = patch_projection(source)
    # Establish a sequential reference for G1. Some graft calls set the
    # launch attribute directly, bypassing the top-level PDL switch.
    source = replace(source,'\nPDL = True\n','\nPDL = False\n')
    source = replace(source,'launch_pdl=True','launch_pdl=False',14)
    source, count = re.subn(r'(programmaticStreamSerializationAllowed\s*=\s*)1;',r'\g<1>0;',source)
    assert count == (7 if experts == 'owned' else 5), count
    if experts == 'native':
        # The B1X wrapper only forwards calls once its experiment is disabled.
        # Removing the whole wrapper also removes its stale host weight pointer.
        start = source.index('GLE_B1X = False')
        end = source.index('_pin_parent_prepare = prepare', start)
        suffix = source[end:]
        assert not re.search(r'\b(?:_GleB1x|_gle_b1x_\w+|_GLE_W_CUR|GLE_B1X)\b', suffix)
        source = source[:start] + suffix
        source = header+'\n'+strip_python_comments(source+'\n'+HOST_BRIDGE+module_source(owned))
        selection = 'state["_fork_moe_status"]'
        name = 'G1b'
    else:
        name = 'G1a'
        selection = '"owned" if state["ma_ok"] else "Marlin"'
        source = patch_owned_moe(source, owned)
        # Valid weights outside the owned packing domain use the baseline Marlin
        # bank. Choose it at prepare time and keep the native path unavailable.
        source = replace(source,"if not _fork_install_owned_moe(weights,st):raise RuntimeError('owned expert scales unsupported')",
                         "_fork_install_owned_moe(weights,st)\n    st['fallback_gamma']=(1+weights['post_expert_norm'].float()).to(torch.bfloat16)")
        source = replace(source,'    st["bank"] = None\n','    st["bank"] = _marlin_bank(weights) if st["owned_moe"] is None else None\n')
        source = replace(source,'    st["ma_ok"] = True','    st["ma_ok"] = st["owned_moe"] is not None')
        # An upstream assignment in the FA3 import block clears the Marlin bank.
        line = next(l for l in source.splitlines() if l.startswith('        st["bank"] = None'))
        source = replace(source,line+'\n','')
        start=source.index('    m = torch.empty((num_tokens, size_k)',source.index('def _marlin_experts('))
        stop=source.index('    return m',start)+len('    return m')
        source=source[:start]+'''    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum
    return fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=state["fallback_gamma"], eps=1e-8,
                                                weight_first=True, slot_expert_ids=local_ids)'''+source[stop:]
    # Removing Python comments from the published file also removes
    # the Python header. Restore the entire upstream attribution with our notice.
    upstream_header=raw.decode('utf-8').split('"""',1)[0]
    source=upstream_header+f'# Modified by yelgabo: {name}, {experts} MoE, repaired precision and ordered layer dispatch.\n' +source.split('\n',1)[1]
    source += f'''
_g1a_prepare=prepare
def prepare(weights):
    state=_g1a_prepare(weights)
    print("{name} backend", {selection},
          "Lt plans", sorted(state.get("graft_lt", {{}})), flush=True)
    return state
'''
    source += '\n'+'\n'.join('# '+line for line in notices.splitlines())+'\n'
    compile(source,'<attributed layer>','exec')
    verify_gle_generator(source)
    if experts == 'native':
        functions = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        for name, references in {
            '_ma_prepare': {'_fork_install_owned_moe', '_fork_native_ma_prepare', '_fork_native_moe'},
            '_moe': {'_fork_moe_fallback', '_fork_native_moe'},
        }.items():
            calls = {n.func.id for n in ast.walk(functions[name]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            if not references <= calls or not references <= functions.keys():
                raise ValueError(f'{name} lost its native/fallback host bridge')
    return source.encode('utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--experts', choices=('owned','native'), default='owned')
    args=parser.parse_args()
    repo = args.upstream.parent.parent
    notices = '\n'.join((repo/name).read_text(encoding='utf-8') for name in ('LICENSE','NOTICE'))
    owned=(Path(__file__).resolve().parents[1]/'muse_decode_layer/cand/layer_attn31s.py').read_text(encoding='utf-8')
    payload=build(args.upstream.read_bytes(), notices, owned, args.experts)
    if len(payload)>=1_000_000:
        raise ValueError(f'payload exceeds judge limit ({len(payload)} bytes)')
    args.output.write_bytes(payload)
    print(len(payload), hashlib.sha256(payload).hexdigest())
