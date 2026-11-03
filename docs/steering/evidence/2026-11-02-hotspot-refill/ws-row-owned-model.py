"""CPU layout/work model only; this does not compile or execute a CUDA candidate."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
import numpy as np

UNIT, CODES = 36864, 32768

def check(root: Path) -> dict:
    source = root / 'muse_decode_layer/cand/g1b4_stage_release.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
    old = np.arange(4 * UNIT, dtype=np.int64).reshape(4, UNIT)
    new = np.full((4, UNIT), -1, dtype=np.int64)
    for stage in range(4):
        for warp in range(4):
            for half in range(2):
                old_k = 2 * stage + half
                new[stage, warp*8192+half*4096:warp*8192+(half+1)*4096] = old[warp, old_k*4096:(old_k+1)*4096]
                new[stage, CODES+warp*1024+half*512:CODES+warp*1024+(half+1)*512] = old[warp, CODES+old_k*512:CODES+(old_k+1)*512]
    assert np.array_equal(np.sort(new.ravel()), np.arange(4 * UNIT))
    restored = np.empty_like(old)
    restored.ravel()[new.ravel()] = np.arange(4 * UNIT)
    assert np.array_equal(new.ravel()[restored.ravel()], np.arange(4 * UNIT))
    for stage in range(4):
        for warp in range(4):
            for chunk in range(16):
                for lane in range(32):
                    a = warp*8192 + chunk*512 + lane*16
                    h, c = divmod(chunk, 8)
                    b = (2*stage+h)*4096 + c*512 + lane*16
                    assert np.array_equal(new[stage,a:a+16], old[warp,b:b+16])
    partition_cases = 0
    min_active_b = {}
    for jobs in range(89):
        credits = np.zeros(jobs, dtype=np.int64)
        for blocks, old_per_new in ((128,4),(64,4)):
            seen = np.zeros(jobs*blocks, dtype=np.int64)
            for cta in range(132):
                lo, hi = jobs*blocks*cta//132, jobs*blocks*(cta+1)//132
                seen[lo:hi] += 1
                if blocks == 128:
                    for u in range(lo,hi):
                        credits[u//blocks] += old_per_new
            assert np.all(seen == 1)
            partition_cases += 1
        assert np.all(credits == 512)
        if jobs in (0,1,2,3,4,8,16,32):
            min_active_b[str(jobs)] = sum(jobs*64*(c+1)//132 > jobs*64*c//132 for c in range(132))
    archive = root / 'docs/steering/evidence/2026-11-02-gopt-skinny-depth'
    sass = {}
    for name, first, last in (('tiny2',0x112c0,0x113e0),('mid',0x10af0,0x10c00)):
        raw = gzip.decompress((archive / f'{name}-control.sass.gz').read_bytes())
        text = raw.decode('utf-8')
        entries = text.split('Function :')[1:]
        counts = []
        for entry in entries:
            sites=[]
            for line in entry.splitlines():
                match=re.search(r'/\*([0-9a-f]+)\*/',line)
                if match and first<=int(match[1],16)<=last and 'LDS.128' in line:
                    sites.append(match[1])
            if sites:
                assert len(sites)==8
                counts.append(sites)
        assert len(counts)==2
        sass[name]={'sha256':hashlib.sha256(raw).hexdigest(),'helper_partial_load_sites_by_entry':counts}
    scores={}
    for r in (0.03,0.05,0.10):
        scores[str(r)]={'equal_saving_fraction_T8_T16':r,'six_factor':(1/(1-r))**(2/6),'elapsed_reduction':1-(1-r)**(2/6)}
    for delta in (1,2,3):
        ratio=((79.112/(79.112-delta))*(117.016/(117.016-delta)))**(1/6)
        scores[f'{delta}us_each']={'six_factor':ratio,'elapsed_reduction':1-1/ratio}
    return {
        'scope':'CPU source/layout/work checks; no native candidate, new compiler invocation, GPU result or measured opportunity',
        'source_sha256':digest,
        'release_sha256':'5df257afd6ae2863bc706826a9798d97b17e73830f1d10703c81bba21203eb9e',
        'release_manifest_helper_sha256':'b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716',
        'byte_permutation_checked':4*UNIT,
        'code_vector_mapping_checks':4*4*16*32,
        'partition_cases':partition_cases,
        'phase_b_ctas_with_work_by_job_count':min_active_b,
        'weight_bytes_per_job_both_designs':768*UNIT,
        'mma_warp_instructions_per_job_both_designs':768*8*32,
        'baseline_partial_write_plus_read_bytes_per_job':768*2*8*32*4*4,
        'new_generic_activation_load_bytes_per_job_at_8_routes':192*4*8*1024*2,
        'new_generic_activation_load_bytes_per_job_at_1_route':192*4*1*1024*2,
        'old_selected_compute_warps':8,'new_compute_warps':4,'new_threads_with_producer':160,
        'weight_ring_bytes':3*UNIT,'one_activation_tile_bytes':8*1032*2,
        'prospective_smem_cap':160*1024,
        'archived_binary_evidence':sass,'conditional_score_sensitivity':scores,
        'two_load_equal_reduction_to_historical_target':1-(5559.8/5788.720720)**3,
    }

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=check(args.root)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
