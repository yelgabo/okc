"""Private-only overrides for the audited whole-layer four-owner machinery."""
OVERRIDES = r'''
def _fork_native_moe(M=None,state=None):
    ext=_WL_NATIVE(M,state)
    if state is not None and state.get('_wl_probe',False):
        assert state['_fork_native_eligible'] and state['ma_ok'] and not _pdl(M),'native/PDL drift'
        key='tiny' if M<=4 else 'tiny2' if M<=8 else 'mid' if M<=16 else 'b4' if M<=32 else 'main'
        flags=_WL_EXT_FLAGS[id(ext._module)]
        assert ('-DOKC_SHORT_FINAL_RELAXED=1' in flags)==(bool(state['_srm_on']) and key in ('tiny','tiny2')),'short RMW flag drift'
        assert '-DOKC_STAGE_RETIRE=0' not in flags,'stage policy drift'
        if key in ('main','b4'):assert ('-DOKC_B4FIN=1' in flags)==(key=='b4'),'B4 drift'
        assert ext is state['ma_ext'][key],'native cache drift'
        state['_wl_hits'].append((M,key,bool(state['_srm_on'])))
    return ext

def _wl_judged_state(production,unused):
    # Normal judge always receives the unchanged607 control modules.
    assert not production.get('_srm_on',False)
    for ext in production['ma_ext'].values():
        assert '-DOKC_SHORT_FINAL_RELAXED=1' not in _WL_EXT_FLAGS[id(ext._module)]
    return production

def _wl_empty(arms):
    checked=0
    for t in (1,8):
        for arm in arms:
            dev=arm['qkv'].device
            x=torch.randn((t,4096),device=dev,dtype=torch.bfloat16)
            ids=torch.full((t,8),-1,device=dev,dtype=torch.int32)
            weights=torch.full((t,8),.125,device=dev,dtype=torch.float32)
            out=torch.full_like(x,17)
            ext=_fork_native_moe(t,arm)
            def call():
                out.fill_(17)
                ext.okc_run(arm['ma_w'],arm['ma_scratch'],x,ids,weights,out,0,32,arm['nsm'])
            call();torch.cuda.synchronize()
            assert bool(torch.isfinite(out).all()) and int(torch.count_nonzero(out))==0,'SRM_EMPTY_EAGER'
            checked+=1
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):call()
            for _ in range(3):
                graph.replay();torch.cuda.synchronize()
                assert bool(torch.isfinite(out).all()) and int(torch.count_nonzero(out))==0,'SRM_EMPTY_GRAPH'
                checked+=1
            # No manual scratch/counter reset; occupied full-model tests follow.
    assert checked==32
    return checked

def _srm_resource_gate(arms,budget):
    result={}
    for key,t in (('tiny',1),('tiny2',8),('mid',16),('b4',32),('main',128)):
        vectors=[]
        for arm in arms:
            ext=_fork_native_moe(t,arm)
            torch.cuda.synchronize()
            r=list(ext.wl_resources());budget()
            assert len(r)==12 and min(r[4],r[10])>=1,'SRM_RESOURCE'
            flags=_WL_EXT_FLAGS[id(ext._module)]
            assert ('-DOKC_SHORT_FINAL_RELAXED=1' in flags)==(arm['_srm_on'] and key in ('tiny','tiny2')),'SRM_PREP_FLAGS'
            vectors.append(r)
        for i in (1,2,3):
            assert vectors[i]==vectors[0],'SRM_RESOURCE_ATTR_ADMISSION_DRIFT'
        result[key]=vectors
    for key in ('tiny','tiny2'):
        assert arms[0]['ma_ext'][key]._module is arms[1]['ma_ext'][key]._module
        assert arms[2]['ma_ext'][key]._module is arms[3]['ma_ext'][key]._module
        assert arms[0]['ma_ext'][key]._module is not arms[2]['ma_ext'][key]._module
    for key in ('mid','b4','main'):
        assert all(arm['ma_ext'][key]._module is arms[0]['ma_ext'][key]._module for arm in arms),'unaffected module drift'
    return result
'''
