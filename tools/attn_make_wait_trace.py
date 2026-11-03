"""Build full16/32 native owned wait/copy timeline; no dispatch, no public candidate."""
import argparse,ast,hashlib,importlib.util,sys,tempfile
from pathlib import Path
from attn_wait_trace_patch import patch,once

def make(control,output,root):
    src=control.read_text(encoding='utf8');tree=ast.parse(src)
    n=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_ONE_SRC')
    cuda=ast.literal_eval(n.value)
    assert '#define ONE_XPIPE 0' in cuda and '#define ONE_PBULK 0' in cuda and 'constexpr int PF = 3;' in cuda
    assert cuda.count('acc[e] *= inv;')==2
    cuda=cuda.replace('acc[e] *= inv;','acc[e]=__bfloat162float(__float2bfloat16_rn(acc[e]*inv));')
    cuda=patch(cuda)
    cuda+=r"""
std::vector<int64_t> attn_wait_resources(){
 std::vector<int64_t> out;
#define WR(L,M) {Instance<L,M>::prepare();cudaFuncAttributes a{};TORCH_CHECK(cudaFuncGetAttributes(&a,attn_one_kernel<L,M>)==cudaSuccess,"wait attrs"); for(int64_t v:{int64_t(L),int64_t(M),int64_t(a.numRegs),int64_t(a.localSizeBytes),int64_t(Instance<L,M>::BYTES),int64_t(Instance<L,M>::NT),int64_t(Instance<L,M>::per_sm)})out.push_back(v);}
 ONE_INSTANCES(WR)
#undef WR
 return out;
}
"""
    cpp=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_ONE_CPP')
    src=once(src,ast.get_source_segment(src,cpp), '_ONE_CPP = '+repr(ast.literal_eval(cpp.value)+'\nstd::vector<int64_t> attn_wait_resources();\n'))
    tree=ast.parse(src);n=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_ONE_SRC')
    lines=src.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['_ONE_SRC = '+repr(cuda)+'\n'];src=''.join(lines)
    spec=importlib.util.spec_from_file_location('_wait_ab',root/'tools/attn_make_bench_ab.py');ab=importlib.util.module_from_spec(spec);spec.loader.exec_module(ab)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d);(p/'muse_attention_decode').mkdir();(p/'muse_attention_decode/submission.py').write_bytes(src.encode())
        old=ab.ROOT;argv=sys.argv
        try:
            ab.ROOT=p;sys.argv=['bench',str(output),'--null','--shapes','full','--inputs','3','--rounds','64','--seed','511019',
                'detailA@own=-DONE_WAIT_TRACE=1,-DONE_WAIT_ONLY=1','detailB@own=-DONE_WAIT_TRACE=1,-DONE_WAIT_ONLY=1,-DWAIT_NULL=1'];ab.main()
        finally:ab.ROOT=old;sys.argv=argv
    out=output.read_text(encoding='utf8')
    out=once(out,'        yield batch, layer, q_len','        if batch not in (16,32):continue\n        yield batch, layer, q_len')
    out=once(out,'    shapes += [_OB_TESTS[i] for i in (0, 1, 2, 3, 7, 9, 10)]','    # Only the four full16/32 cells in pruned diagnostic builds.')
    out=once(out,'_ST_BASE, _ST_W = 1024, 16','_ST_BASE, _ST_W = 1024, 56')
    out=once(out,'data = _ab_shape_inputs(batch, layer, q_len, 1000 + k)','data = _ab_shape_inputs(batch, layer, q_len, 511019 + k)')
    out=once(out,'timing_seeds={[1000 + k','timing_seeds={[511019 + k')
    out=once(out,'def _ab_graph(fn, data, calls):','def _ab_graph(fn, data, calls):\n    data={k:v.clone() if isinstance(v,torch.Tensor) else dict(v) if isinstance(v,dict) else v for k,v in data.items()}')
    out=once(out,'    return graph\n','    graph._wait_inputs=data  # retain every private input/cache backing allocation\n    return graph\n')
    target='functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop"]'
    assert out.count(target)==2
    out=out.replace(target,target[:-1]+', "attn_wait_resources"]')
    body=Path(__file__).with_name('attn_wait_trace_body.py').read_text(encoding='utf8')
    out=once(out,'def _ab_check():',body+'\n\ndef _ab_check():\n    global _WAIT_FAILED\n    _WAIT_FAILED=True\n    _wait_resources()')
    out=once(out,'                    ok, msg = rel < 2e-2, ""','                    ok,msg=_ob_check(data,out)\n                    ok=ok and rel<2e-2')
    out=once(out,'    print(f"ABCHECK done: {fails} failing {_ob_elapsed()}", flush=True)','    assert fails==0,"wait trace full math/cache/row gate"\n    _wait_collect()\n    _WAIT_FAILED=False\n    print(f"ABCHECK done: {fails} failing {_ob_elapsed()}", flush=True)')
    out=once(out,'def _ab_bench(warm_calls=16, warm_reps=25):','def _ab_bench(warm_calls=16, warm_reps=25):\n    assert not _WAIT_FAILED,"no timing after incomplete trace"')
    out+='\nassert not _WAIT_FAILED,"wait diagnostic incomplete"\n_wait_final()\n'
    compile(out,str(output),'exec');output.write_bytes(out.encode());output.with_suffix('.cu').write_bytes(cuda.encode())
    print('WAIT_TRACE',len(out.encode()),hashlib.sha256(out.encode()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--root',type=Path,required=True);a=p.parse_args();make(a.root/'muse_attention_decode/submission.py',a.output,a.root)
