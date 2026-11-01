"""Offline extraction of proven layer MoE as an optional fork fallback.

No job dispatch. No compressed runtime source. Caller explicitly installs/selects
the adapter; native backend policy and Marlin remain the coordinator's decision.
"""
import ast
import hashlib
import re
from dl_make_bench_pub import strip_comments

RUNTIME=r'''
class _OwnedLayerMoe:
    def __init__(self, module, packed, gamma, scratch, zero):
        self.module,self.packed,self.gamma,self.scratch,self.zero=module,packed,gamma,scratch,zero

    def okc_set_pdl(self, value):
        # Always an ordinary ordered launch. No overlapping resident global barrier.
        pass

    def __deepcopy__(self,memo):
        import copy
        result=type(self)(self.module,copy.deepcopy(self.packed,memo),
            copy.deepcopy(self.gamma,memo),copy.deepcopy(self.scratch,memo),copy.deepcopy(self.zero,memo))
        memo[id(self)]=result
        return result

    def okc_run(self, ignored_w, ignored_scratch, x, lids, tw, out, lo, nloc, nsm):
        assert lo==0 and nloc==32 and 1<=len(x)<=384
        assert x.dtype==torch.bfloat16 and lids.dtype==torch.int32 and tw.dtype==torch.float32
        assert out.dtype==torch.bfloat16 and tuple(out.shape)==tuple(x.shape)
        w,s=self.packed,self.scratch
        value=self.module.moe_fused_run(x,lids,tw,self.zero,w['w13'],w['w2'],w['w13_gs'],w['w2_gs'],
            self.gamma,s['inter'],s['ybuf'],s['ss'],s['ctr'],s['done_a'],w['w13f'],w['w2f'],
            s['xa'],s['xb'],s['fix'],s['fixcnt'],s['xready'],s['done_as'],32,1e-8,
            0 if len(x)<=16 else 1,False)
        out.copy_(value)

def _fork_install_owned_moe(weights,state):
    # Explicit opt-in by coordinator; no implicit policy substitution on a compile error.
    packed=_owned_moe1_prepare(weights)
    if packed is None:
        state['owned_moe']=None
        return False
    module=load_inline(name='okc_fork_owned_moe_v1',cpp_sources=_OWNED_MOE_CPP,
        cuda_sources=_OWNED_MOE_DEVICE+'\n'+_OWNED_MOE_HOST,functions=['moe_fused_run'],
        extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],verbose=False)
    dev=weights['w13'].device
    # Exact layer ABI: effective gamma is BF16-rounded before conversion to FP32.
    gamma=(1.0+weights['post_expert_norm'].float()).to(torch.bfloat16).float().contiguous()
    state['owned_moe']=_OwnedLayerMoe(module,packed,gamma,_owned_moe1_scratch(dev),
                                    torch.zeros(1,dtype=torch.int32,device=dev))
    return True

def _fork_owned_moe_for(M,state):
    # The original layer chunks global T to <=384 before okc_run.
    return state.get('owned_moe') if state is not None and M is not None and M>=1 else None
'''

def module_source(owned):
    assert hashlib.sha256(owned.encode()).hexdigest()=='585d46e2cd059ea7ea76c25110fd199bb228d2ee7f96239b44ae0bfb8025893b', 'unreviewed owned source'
    tree=ast.parse(owned)
    vals={t.id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
          for t in n.targets if isinstance(t,ast.Name) and t.id in
          ('_MOE1_SRC','_MOE1_HOST_SRC','_MOE1_CPP_SRC','W4_BITPOS')}
    device=vals['_MOE1_SRC']
    for ns in ('sk_8_4_0','sk_16_4_0','sk_8_4_3'):
        a=device.index('namespace '+ns+' {')
        z=device.index('}  // namespace '+ns,a)+len('}  // namespace '+ns)
        device=device[:a]+device[z:]
    assert device.count('namespace sk_8_3_0 {')==1
    host=vals['_MOE1_HOST_SRC'].replace('//@MOE1_DISPATCH@',
        'case 0: sk_8_3_0::launch<1>(p,sms,stream); break;\ncase 1: launch_main<51>(p,sms,stream); break;')
    names=('moe1_unit_rows','moe1_repack_expert','moe1_repack_expert_fat',
           '_to_u8_words','_cat_last','_perm','_stack_last','_to_i64','_w4_scales_ok',
           '_moe1_scratch','_moe1_prepare')
    helpers='\n\n'.join(ast.get_source_segment(owned,n) for n in tree.body
                        if isinstance(n,ast.FunctionDef) and n.name in names)
    assert len([n for n in ast.parse(helpers).body if isinstance(n,ast.FunctionDef)])==len(names)
    mapping={n:('_owned'+n if n.startswith('_') else '_owned_'+n) for n in names}
    mapping.update(W4_BITPOS='_OWNED_W4_BITPOS',TOP_K='_OWNED_TOP_K',LOCAL_EXPERTS='_OWNED_LOCAL_EXPERTS',
        D_EXPERT='_OWNED_D_EXPERT',_MOE1_EMAX='_OWNED_EMAX',_MOE1_JMAX='_OWNED_JMAX',
        _MOE1_JMAX_MAIN='_OWNED_JMAX_MAIN')
    helpers=re.sub(r'\b('+'|'.join(re.escape(n) for n in mapping)+r')\b',lambda m:mapping[m[0]],helpers)
    out=('\n_OWNED_MOE_DEVICE='+repr(device)+'\n_OWNED_MOE_HOST='+repr(host)
         +'\n_OWNED_MOE_CPP='+repr(vals['_MOE1_CPP_SRC'])
         +'\n_OWNED_W4_BITPOS='+repr(vals['W4_BITPOS'])
         +'\n_OWNED_TOP_K=8\n_OWNED_LOCAL_EXPERTS=32\n_OWNED_D_EXPERT=4096\n'
         +'_OWNED_EMAX=4096\n_OWNED_JMAX=544\n_OWNED_JMAX_MAIN=96\n'+helpers+'\n'+RUNTIME)
    ast.parse(out)
    return out

def patch_owned_moe(source, owned):
    """Return a fully wired owned-only expert backend with no donor expert launches.

    Whole-line comments are removed by the repository's established safe stripper;
    attribution/NOTICE must be appended by the coordinator after this composition.
    """
    # Strip all reachable native build/repack entry points. Keep native source strings
    # inert for now; deleting import-time string patches would introduce new dependencies.
    tree=ast.parse(source);raw=source.encode();lines=raw.splitlines(keepends=True);offset=[0]
    for line in lines:offset.append(offset[-1]+len(line))
    replacement={
        '_ma_ext':'def _ma_ext(*args,**kwargs):\n    raise RuntimeError("donor expert builds disabled in owned-only fork")\n',
        '_moe_build':'def _moe_build(*args,**kwargs):\n    raise RuntimeError("donor expert builds disabled in owned-only fork")\n',
        '_moe_mid':'def _moe_mid():\n    raise RuntimeError("native prewarm disabled")\n',
        '_moe_tiny':'def _moe_tiny():\n    raise RuntimeError("native prewarm disabled")\n',
        '_moe_tiny2':'def _moe_tiny2():\n    raise RuntimeError("native prewarm disabled")\n',
        '_moe':'''def _moe(M=None,state=None):
    result=_fork_owned_moe_for(M,state)
    if result is None:raise RuntimeError('owned experts require prepared per-state adapter')
    return result
''',
        '_ma_prepare':'''def _ma_prepare(weights,st):
    if not _fork_install_owned_moe(weights,st):raise RuntimeError('owned expert scales unsupported')
    st['ma_w']=()
    st['ma_scratch']=()
'''}
    edits=[];counts={}
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in replacement:
            edits.append((offset[n.lineno-1]+n.col_offset,offset[n.end_lineno-1]+n.end_col_offset,replacement[n.name].encode()))
            counts[n.name]=counts.get(n.name,0)+1
    assert counts=={'_ma_ext':1,'_moe_build':1,'_moe':2,'_moe_mid':1,'_moe_tiny2':1,'_moe_tiny':1,'_ma_prepare':1},counts
    for a,z,value in sorted(edits,reverse=True):raw=raw[:a]+value+raw[z:]
    source=raw.decode()
    # No native prewarms, scale-codegen specialization, legacy expert repack or gamma blob.
    line=next(line for line in source.splitlines() if line.lstrip().startswith('ths = [threading.Thread'))
    assert 'lambda: _moe(32)' in line
    source=source.replace(line,'    ths = [threading.Thread(target=_build, args=(f,)) for f in (_att7, bfp__get_ext, _graft_glue_module, _graft_b2_module, _sl_cluster_module)]')
    for prefix in ('    _MA_SFX = _MA_SFX_IF_PRMT','    _MA_SC6W = bool('):
        line=next(line for line in source.splitlines() if line.startswith(prefix))
        source=source.replace(line,'    _MA_SFX = ()' if '_MA_SFX =' in line else '    _MA_SC6W = False')
    for key in ('w13q','w13s','w2q','w2s','gs13','gs2','g_postexp'):
        lines=[line for line in source.splitlines(keepends=True) if line.startswith('    st["'+key+'"] =')]
        assert len(lines)==1,(key,len(lines))
        source=source.replace(lines[0],'')
    prewarm='    for T in (1, 8, 16, 32, 128, 256):\n        _moe(T, st)\n'
    assert source.count(prewarm)==1
    source=source.replace(prewarm,'')
    header=source.splitlines()[0]
    out=strip_comments(source+module_source(owned))
    if header.startswith('# okc:'):out=header+'\n'+out
    ast.parse(out)
    if len(out.encode())>=1_000_000:raise ValueError(f'owned fallback payload {len(out.encode())} exceeds genuine limit')
    return out
