"""Conservative projection repairs for pinned keell0renz decode-layer fork.

Retain its attribution/NOTICE. Root composes front/boundary repairs separately.
No rs128 BF16 split reduction or calibration kernel is executed after this patch.
"""
import ast

LT_GUARD='''
def _fork_lt_no_split(lt, algo):
    # CUBLASLT_ALGO_CONFIG_SPLITK_NUM=2. Reject all K-split opaque plans;
    # output-feature batching in _graft_layouts still performs the full K dot product.
    value, written = _GI(), _GSZ()
    lt.call("cublasLtMatmulAlgoConfigGetAttribute", _C.byref(algo), 2,
            _C.byref(value), _C.sizeof(value), _C.byref(written))
    return written.value == _C.sizeof(value) and value.value in (0, 1)

'''

def patch_projection(source: str) -> str:
    # Entire tail is rs128 plus optional MFAB calibration wrappers. Only _rq_gemm
    # and the pre-existing prepare name are referenced above this cut in pinned source.
    cut=source.index('def _make_chen():')
    before,after=source[:cut],source[cut:]
    later={n.name for n in ast.parse(after).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    referenced={n.id for n in ast.walk(ast.parse(before)) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
    assert later & referenced <= {'prepare','_rq_gemm'}, later & referenced
    assert '_rq_time' not in before and '_CHEN' not in before
    source=before+'''def _rq_gemm(state, name, x, T):
    # Fork precision repair: BF16-rounded split-K partial reductions disabled.
    # Existing checked cuBLASLt plans handle the active 128/256 routes instead.
    return None
'''
    # Only edit the ordinary sk2 template; no approximate diagnostics or barrier deletions.
    start=source.index('bfp_SK2_SRC = r"""')
    end=source.index('"""',start+len('bfp_SK2_SRC = r"""'))
    block=source[start:end]
    old='if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");'
    assert block.count(old)==2
    # Both issuing paths: CL>1 tid0 before initial x TMA; CL1 all threads after wait,
    # including tid0 and the refill producer. Frozen weight TMA prefetch stays early.
    lines=block.splitlines(keepends=True)
    patched=[]
    for line in lines:
        patched.append(line)
        if old in line:
            indent=line[:len(line)-len(line.lstrip())]
            patched.append(indent+'if (pdl) asm volatile("fence.proxy.async.global;" ::: "memory");  // upstream generic x -> TMA\n')
    source=source[:start]+''.join(patched)+source[end:]
    # Existing descriptor requests FP32 compute (68), BF16 inputs and BF16 dense outputs.
    # Opaque saved algo words are accepted only if the library confirms no K split.
    signature='            "cublasLtMatmulAlgoCheck":'
    assert source.count(signature)==1
    source=source.replace(signature,
        '            "cublasLtMatmulAlgoConfigGetAttribute": [_C.POINTER(_GraftAlgo), _GI, _GP, _GSZ, _C.POINTER(_GSZ)],\n'+signature)
    anchor='def _graft_lt_plan(st):'
    assert source.count(anchor)==1
    source=source.replace(anchor,LT_GUARD+anchor)
    line='        plans[(name, m)] = _GraftPlan('
    assert source.count(line)==1
    source=source.replace(line,'        if not _fork_lt_no_split(lt, algo):\n            status[(name,m)] = ("rejected K split", 0)\n            continue\n'+line)
    line='        if not code and heur.workspaceSize <= lt_tn.workspace.numel():'
    assert source.count(line)==1
    source=source.replace(line,'        if not code and heur.workspaceSize <= lt_tn.workspace.numel() and _fork_lt_no_split(lt_tn, algo):')
    # Preserve sk2 generic FP32 correction/counter joins and all shared/cluster barriers.
    assert source.count('atom.add.release.gpu.global.s32')==before.count('atom.add.release.gpu.global.s32')
    assert source.count('fence.acq_rel.gpu;')==before.count('fence.acq_rel.gpu;')
    ast.parse(source)
    return source
