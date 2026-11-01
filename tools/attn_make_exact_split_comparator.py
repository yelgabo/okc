"""Offline CHECK-ONLY exact public split-engine comparator; never a submission."""
import argparse
import hashlib
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import attn_make_bench_ab as ab


def once(s, old, new):
    assert s.count(old) == 1, old[:100]
    return s.replace(old, new, 1)


def build(control, donor, output):
    raw = donor.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '496d460e0bb1e7489684de2742f1ff22f3959d9d5fe30674199e74b3d002d651'
    src = raw.decode().replace('\r\n', '\n')
    assert src.count('"-DZH_SCH=1"') == 2
    src = src.replace('"-DZH_SCH=1"', '"-DZH_SCH=0"')
    # Host-only resource gate/metadata; no device math or synchronization edits.
    src = once(src, '    cudaLaunchConfig_t cfg = {};\n    cfg.gridDim = dim3((unsigned)G, 1, 1);', '''    static bool reported=false;
    if(!reported){cudaFuncAttributes a{}; int resident=0;
    TORCH_CHECK(cudaFuncGetAttributes(&a, f0)==cudaSuccess, "resource attrs");
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,f0,256+128*NPW,smem)==cudaSuccess && resident>=1, "resource residency");
    TORCH_CHECK(a.localSizeBytes==0, "WST spill/local memory gate");
    printf("KEL_RESOURCE WST regs=%d local=%zu dynamic=%d blocksSM=%d\\n",a.numRegs,a.localSizeBytes,smem,resident);fflush(stdout);reported=true;}
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)G, 1, 1);''')
    src = once(src, '  cudaError_t e = cudaLaunchKernelEx(&cfg, vp2_kernel,', '''  static bool reported=false;
  if(!reported){cudaFuncAttributes a{};int clusters=0;
    TORCH_CHECK(cudaFuncGetAttributes(&a,vp2_kernel)==cudaSuccess,"VP2 attrs");
    TORCH_CHECK(cudaOccupancyMaxActiveClusters(&clusters,vp2_kernel,&cfg)==cudaSuccess && clusters>=2,"VP2 cluster residency");
    TORCH_CHECK(a.localSizeBytes==0,"VP2 spill/local memory gate");
    printf("KEL_RESOURCE VP2 regs=%d local=%zu dynamic=%d activeClusters=%d\\n",a.numRegs,a.localSizeBytes,smem,clusters);fflush(stdout);reported=true;}
  cudaError_t e = cudaLaunchKernelEx(&cfg, vp2_kernel,''')
    # Keep donor outside git, embed into diagnostic only.
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'muse_attention_decode').mkdir()
        (root/'muse_attention_decode/submission.py').write_text(control.read_text(encoding='utf-8'),encoding='utf-8')
        private=root/'exact_public.py';private.write_text(src,encoding='utf-8')
        oldroot,oldargv=ab.ROOT,sys.argv
        try:
            ab.ROOT=root
            sys.argv=['attn_make_bench_ab.py',str(output),'--pub',f'pubkel={private}','--null','--shapes','all','--inputs','2','--rounds','100','--seed','311151']
            ab.main()
        finally:ab.ROOT,sys.argv=oldroot,oldargv
    bench=output.read_text(encoding='utf-8')
    bench=once(bench,'def _ab_check():\n    fns = _ab_fns()', '''def _ab_check():
    global _KEL_CHECK_FAILED
    _KEL_CHECK_FAILED = True
    assert set(_AB_PUBS)=={'pubkel','pubkel2'}, 'missing independent public copies'
    for name,mod in _AB_PUBS.items():
        mod._zh_join()
        assert 'wst' in mod._ZH and 'vfg' in mod._ZH, (name,mod._ZH.keys())
        print('KEL_BUILD',name,'SCH=0 native WST full16/VP2 sliding1; independent workspace',flush=True)
    print('KEL_CONTROL_RESOURCE',_ONE.attn_one_prepare(),flush=True)
    fns = _ab_fns()''')
    bench=once(bench,'    for shape in shapes:', '    shapes += [(16,80000,"tp8","full",8,"agentic","coherent",1,1.0),(1,80000,"tp8","sliding",8,"fixed","soft",1,1.0)]  # fresh native banks\n    for shape_index, shape in enumerate(shapes):')
    bench=once(bench,'            data = _ob_inputs(*shape)', '            data = _ob_inputs(*shape, seed=311151+shape_index*997)')
    # Existing official-style two-reference/global/row/cache check for EVERY arm.
    bench=once(bench,'                    ok, msg = rel < 2e-2, ""','                    ok, msg = _ob_check(data, out)\n                    ok = ok and rel < 2e-2')
    bench=once(bench,'    print(f"ABCHECK done: {fails} failing {_ob_elapsed()}", flush=True)', '''    _KEL_CHECK_FAILED = bool(fails)
    assert not _KEL_CHECK_FAILED, 'official-math/cache/row comparator gate'
    print(f"ABCHECK done: {fails} failing {_ob_elapsed()}", flush=True)''')
    bench=once(bench,'def _ab_bench(warm_calls=16, warm_reps=25):','def _ab_bench(warm_calls=16, warm_reps=25):\n    assert not _KEL_CHECK_FAILED, "no timing after failed correctness"')
    a=bench.index('def _ab_check():'); b=bench.index('def _ab_graph(',a)
    segment=bench[a:b]
    assert segment.count('ok, msg = _ob_check(data, out)') == 2
    bench=bench[:a]+segment.replace('ok, msg = _ob_check(data, out)', 'ok, msg = _kel_math_check(data, out)')+bench[b:]
    bench=once(bench, 'def _ab_check():', """def _kel_math_check(data, out):
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return _ob_check(data, out)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _ab_check():""")
    bench += '\nassert not _KEL_CHECK_FAILED, "CHECK ONLY comparator failed; not a candidate"\n'
    compile(bench,str(output),'exec');output.write_text(bench,encoding='utf-8')
    print('EXACT_COMPARATOR_SHA',hashlib.sha256(bench.encode()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True);p.add_argument('--donor',type=Path,required=True);p.add_argument('output',type=Path)
    args=p.parse_args();build(args.control,args.donor,args.output)
