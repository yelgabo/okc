"""Own CPU algebra for legal routes and scheduled work. No donor execution."""
import hashlib
import json
import math
import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
SCRATCH=ROOT/'.build/worker-broad-expert-architecture'
native=(ROOT/'muse_decode_layer/cand/g1b4_stage_release.py').read_text()
assert hashlib.sha256(native.encode()).hexdigest()=='ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
paths={
 'old_scheduler':SCRATCH/'humming-prior/humming/include/humming/scheduler.cuh.txt',
 'new_scheduler':SCRATCH/'humming-source/humming/include/humming/scheduler.cuh.txt',
 'old_ws':SCRATCH/'humming-prior/humming/include/humming/kernel/humming_ws.cuh.txt',
 'new_ws':SCRATCH/'humming-source/humming/include/humming/kernel/humming_ws.cuh.txt',
 'policy':SCRATCH/'humming-source/humming/tune/sm90_policies.py.txt'}
texts={k:p.read_text() for k,p in paths.items()}
assert 'CEIL_DIV(ctx.smem.expert_tokens[index], BlockShape::M)' in texts['old_scheduler']
assert 'uint32_t blocks = CEIL_DIV(tokens, BlockShape::M)' in texts['new_scheduler']
assert 'kUseGroupedRaster' in texts['new_scheduler']
assert 'max_block_m = 64' in texts['policy']
assert 'const int nsj = (tot + SK_T - 1) / SK_T;' in native
assert 'wgs_seg<PB, 8>' in native and 'const int ntt = (S.jC[u >> UBL] + 7) >> 3;' in native
assert '#define OKC_FT_T 64' in native and '#define OKC_SKT 8' in native
ceil=lambda x,n:(x+n-1)//n

def fixture(t,kind):
    ids=[]
    for i in range(t):
        remote=[32+(8*i+j)%224 for j in range(8)]
        if kind=='empty':row=remote
        elif kind=='uniform_one_local':row=[i%32]+remote[:7]
        elif kind=='balanced_eight_local':row=[(8*i+j)%32 for j in range(8)]
        elif kind=='one_hot_local':row=[0]+remote[:7]
        elif kind=='eight_hot_local':row=list(range(8))
        elif kind=='irregular_two_local':row=[0 if i<(t//2+1) else 1]+remote[:7]
        else:raise ValueError(kind)
        assert len(row)==len(set(row))==8 and all(0<=x<256 for x in row)
        ids.append(row)
    counts=[sum(e in row for row in ids) for e in range(32)]
    assert all(c<=t for c in counts) and sum(counts)<=8*t
    return ids,counts

def schedule(counts,bm,n,bk):
    mt=sum(ceil(c,bm) for c in counts)
    kblocks=4096//bk
    nt=n//128
    return dict(token_tiles=mt,output_tiles=mt*nt,k_loop_tile_iterations=mt*nt*kblocks,
        padded_token_rows=mt*bm,normalized_gemm_flops=2*mt*bm*n*4096,
        logical_weight_elements_requested=mt*n*4096,
        geometry=dict(token_m=bm,output_n=128,k=bk))

rows=[]
for t in (16,32,128,256):
 for kind in ('empty','uniform_one_local','balanced_eight_local','one_hot_local','eight_hot_local','irregular_two_local'):
    ids,counts=fixture(t,kind)
    jobs=sum(ceil(c,8 if t==16 else 64) for c in counts)
    padded=8*sum(ceil(c,8) for c in counts)
    control={}
    for name,n in (('gateup',8192),('down',4096)):
        width=16 if t==16 else 256
        control[name]=dict(job_weight_streams=jobs,output_units=jobs*(n//width),
            output_unit_rows=width,logical_k256_weight_pieces=jobs*(n//width)*16,
            padded_token_rows=padded,normalized_gemm_flops=2*padded*n*4096,
            logical_weight_elements_requested=jobs*n*4096,
            useful_gemm_flops=2*sum(counts)*n*4096)
    geometric=[]
    # Symbolic geometry comparisons, not tuning/foreign-policy execution or a
    # claim these are the missing old actual configs. Output/K geometry fixed.
    for bm in (8,16,32,64):
        g={name:schedule(counts,bm,n,64) for name,n in (('gateup',8192),('down',4096))}
        geometric.append(dict(token_m=bm,old_actual_config_identified=False,
            old_masked_at_this_geometry=g,new_contiguous_at_this_geometry=g,
            old_masked_capacity_rows=32*ceil(t,bm)*bm,
            skipped_capacity_rows=32*ceil(t,bm)*bm-g['down']['padded_token_rows'],
            normalized_mma_work_ratio_vs_native=(g['down']['normalized_gemm_flops']/control['down']['normalized_gemm_flops']) if padded else None,
            logical_weight_request_ratio_vs_native=g['down']['logical_weight_elements_requested']/control['down']['logical_weight_elements_requested'] if jobs else None))
    assert geometric[0]['old_masked_at_this_geometry']==geometric[0]['new_contiguous_at_this_geometry']
    assert geometric[0]['new_contiguous_at_this_geometry']['down']['padded_token_rows']==padded
    assert all(x['new_contiguous_at_this_geometry']['down']['padded_token_rows']>=padded for x in geometric)
    rows.append(dict(tokens=t,kind=kind,ids=ids,histogram=counts,valid_local_routes=sum(counts),
        metadata_slots=8*t,valid_silu_elements=sum(counts)*4096,norm_elements=sum(counts)*4096,
        control=control,grouped_geometry_comparisons=geometric))

analysis=json.loads((ROOT/'docs/steering/evidence/2026-11-02-attention-roi-pivot/analysis.json').read_text())
C=analysis['weighted_layer_endpoints_us'];E=analysis['long_weighted_expert_activity_us']
def factor(f,h):return math.exp(sum(math.log(C[k]/(C[k]-f*E[k]+h)) for k in E)/6)-1
scenarios=[dict(gross_fraction=f,incremental_cost_per_load_us=h,layer_gain_if_fully_exposed=factor(f,h)) for f in (0,.1,.2,.3) for h in (0,5,10)]
out=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision='NO SUCH WORK REDUCTION IDENTIFIED',old_runtime_config=None,
    pins={k:dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()},
    source_checks='PASS',fixtures=rows,cost_scenarios=scenarios,
    assumptions=['Fixed legal analytical fixtures, not recovered judge routing. Both GEMMs counted.',
        'Geometric full-K tile work; Stream-K partitions do not remove required K blocks. Tile counts are not instruction counts across architectures.',
        'Logical weight requests do not establish physical traffic or exposed time. Output-store masks do not reduce counted matrix math.',
        'Cost sensitivities assume gross activity fully exposed; 5/10us are assumed NET incremental costs, not measured launch costs.'],
    measured_savings=None,new_check_authorized=False)
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'census.json').write_text(json.dumps(out,indent=2)+'\n')
files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
(OUT/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps(dict(fixtures=len(rows),source_checks=out['source_checks'],decision=out['decision'],scenarios=scenarios),indent=2))
