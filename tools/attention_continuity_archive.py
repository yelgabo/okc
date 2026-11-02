"""Archive the finite offline continuity implementation (no target service)."""
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
from attention_continuity import generate, ROOT, LAYER, sha
from attention_top100_screen import screen


if __name__ == '__main__':
    build = ROOT/'.build/attention-continuity'
    out = ROOT/'docs/steering/evidence/2026-11-02-attention-continuity'
    out.mkdir(parents=True,exist_ok=True)
    for name in ('candidate.py','control.cu','treatment.cu','native.patch','manifest.json'):
        shutil.copyfile(build/name,out/name)
    compilation=[]
    for v in ('control','treatment'):
        for length in (1,8):
            folder=f'{v}-L{length}'
            shutil.copytree(build/folder,out/folder,dirs_exist_ok=True,copy_function=shutil.copyfile)
            m=json.loads((out/folder/'manifest.json').read_text())
            assert m['nvrtc_status']==m['ptxas_status']==0
            expected=(128,3) if length==1 else (80,16)
            assert re.search(r'0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads',m['ptxas_output'])
            assert f'Used {expected[0]} registers, used {expected[1]} barriers' in m['ptxas_output']
            ptx=(out/folder/'device.ptx').read_text()
            raw_layout=re.search(r'continuity_layout\[8\] = \{([^}]+)\}',ptx).group(1)
            layout=bytes([int(x.strip()) for x in raw_layout.split(',')]).ljust(8,b'\0')
            assert (int.from_bytes(layout[:4],'little'),int.from_bytes(layout[4:],'little'))==(61440,113664)
            compilation.append(dict(variant=v,length=length,manifest=folder+'/manifest.json'))
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tools','-p','test_attention_continuity.py','-v'],capture_output=True,text=True,cwd=ROOT)
    (out/'cpu-tests.txt').write_text(result.stdout+result.stderr)
    result.check_returncode()
    baseline=(ROOT/LAYER).read_text()
    routes=screen(baseline)['launch_rows']
    for r in routes:r['module']='continuity' if r['batch'] in (16,32) and r['window']==0 else 'original'
    resources=dict(L1=dict(registers=128,barriers=3,stack=0,spill_store=0,spill_load=0,threads=256,
        sizeof_smem=61440,control_dynamic=62464,treatment_dynamic=67072),
        L8=dict(registers=80,barriers=16,stack=0,spill_store=0,spill_load=0,threads=384,
        sizeof_smem=113664,control_dynamic=114688,treatment_dynamic=114688))
    summary=dict(base_sha256=sha(baseline),candidate_sha256=sha((out/'candidate.py').read_text()),
        launch_rows=routes,compilation=compilation,resources=resources,
        cpu_exitcode=result.returncode,scope='Offline source/CPU/secondary compiler only; target correctness, host build, admission and timing NOT_RUN',
        adapters_initial_failures=['NVRTC rejects NVCC-only --expt-relaxed-constexpr','NVRTC builtins require installed library directory in LD_LIBRARY_PATH','cuda.h requires declaration-only stdlib.h virtual header; device extract already closes namespace; INFINITY declaration supplied'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.rglob('*') if p.is_file() and p.name!='archive-manifest.json'}
    (out/'archive-manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps(dict(files=len(files),candidate=summary['candidate_sha256'],cpu_exitcode=result.returncode)))
