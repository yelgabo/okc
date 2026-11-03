"""Source pins and conditional cost algebra only; no donor import or GPU calls."""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
base=ROOT/'muse_decode_layer/cand/g1b4_stage_release.py'
text=base.read_text(); assert hashlib.sha256(text.encode()).hexdigest()=='ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
parent=json.loads((ROOT/'docs/steering/evidence/2026-11-02-attention-roi-pivot/analysis.json').read_text())
C=parent['weighted_layer_endpoints_us']; E=parent['long_weighted_expert_activity_us']
def gain(frac):return math.exp(sum(math.log(C[k]/(C[k]-frac*E[k])) for k in E)/6)-1
def fraction(target):
    lo,hi=0.,.99
    for _ in range(100):
        mid=(lo+hi)/2
        if gain(mid)<target:lo=mid
        else:hi=mid
    return (lo+hi)/2
budgets=[]
for target in (.03,parent['current_layer_score_ratio_needed']-1,.10):
    f=fraction(target)
    budgets.append(dict(target_layer_gain=target,conditional_expert_activity_fraction=f,
        required_net_weighted_savings_us={k:E[k]*f for k in E}))
per_expert=8192*4096+4096*4096
costs=[]
for t in (16,32,128,256):
    active=32*(1-(31/32)**t)
    packed=active*per_expert*(.5+1/16)
    sc6=active*per_expert*(.5+6/8/16)
    expanded=active*per_expert*2
    costs.append(dict(tokens=t,expected_local_routes=t,expected_tokens_per_expert=t/32,
        expected_active_experts=active,encoded_bytes_single_pass=packed,sc6_bytes_single_pass=sc6,
        expanded_bf16_bytes_single_pass=expanded,
        ideal_expanded_stream_us_at_4p8TBps=expanded/4.8e6,
        ideal_sc6_stream_us_at_4p8TBps=sc6/4.8e6,
        useful_gemm_flops=2*t*per_expert,
        grouped_intermediates_min_read_write_bytes=2*t*2*(4096+8192+4096+4096),
        masked_workspace_cap_T_bytes=2*32*t*(4096+8192+4096+4096)))
nodes=ast.parse(text).body
functions={n.name:dict(line=n.lineno,end=n.end_lineno,sha256=hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest()) for n in nodes if isinstance(n,ast.FunctionDef) and n.name in ('_fork_native_moe','_fork_native_ma_prepare','_sc6_wgs','_sc6_ok','_pack_codes','_wgs_copy','_ma_ext','_after_attention')}
out=dict(base_sha256=hashlib.sha256(text.encode()).hexdigest(),budgets=budgets,costs=costs,
    full_bank=dict(elements=32*per_expert,encoded_bytes=32*per_expert*9/16,sc6_bytes=32*per_expert*35/64,expanded_bf16_bytes=32*per_expert*2),
    inflation_vs_encoded=2/(9/16),inflation_vs_sc6=2/(35/64),control_functions=functions,
    assumptions=['Hypothetical uniform independent per-token top8 across256experts: any named expert inclusion probability1/32. Not reconstruction of measured routing.',
        'One complete W13/W2 read per active expert, no extra tile reloads. Bytes are logical work; physical HBM/L2 traffic unmeasured.',
        '4.8TB/s is vendor H200 peak, not measured achieved bandwidth. Stream times are conditional optimistic floors if all logical bytes traverse HBM.',
        'Activity fraction model assumes all removed activity exposed and every replacement cost paid; required savings, not a speed prediction.'],
    decision='NO_ADEQUATE_PATH_FOUND',measured_saving=None)
(OUT/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(budgets=budgets,full_bank=out['full_bank'],costs=costs),indent=2))
