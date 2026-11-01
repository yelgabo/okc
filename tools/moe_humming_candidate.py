"""Generate standalone MoE adapter from reviewed Humming indexed prefill BODY.

Usage: python tools/moe_humming_candidate.py PREFILL_GENERATOR MOE_BASELINE OUT
No dispatch; no lowered precision. Supports all complete EP32 banks supplied in weights.
"""
import ast
from pathlib import Path
import sys

ADAPTER = '''
def prepare(weights):
    assert weights["w13"].shape[0] % 32 == 0
    assert weights["w13"].shape[1:] == (8192,2048)
    banks = {}
    for start in range(0,weights["w13"].shape[0],32):
        local = {k:weights[k][start:start+32].contiguous()
                 for k in ("w13","w13_sf","w13_gs","w2","w2_sf","w2_gs")}
        banks[start] = _hi_prepare(local)
    return {"banks":banks,"post_expert_w":(weights["post_norm_w"].float()+1).to(torch.bfloat16)}

def kernel_fn(data,state):
    cfg = data["config"]
    assert cfg["local_count"] == 32
    start = cfg["local_start"]
    sub = {"hum":state["banks"][start],"post_expert_w":state["post_expert_w"]}
    local = data["topk_ids"].to(torch.int32).contiguous() - start
    return _experts(data["hidden_states"].contiguous(),
                    data["topk_weights"].float().contiguous(),local,sub)
'''

def main():
    gen, baseline, out = map(Path,sys.argv[1:])
    body = next(ast.literal_eval(n.value) for n in ast.parse(gen.read_text()).body
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BODY' for t in n.targets))
    source = '# okc: problem=muse_moe_w4a16\n# Candidate: exact indexed Humming BF16/NVFP4 stages.\n'
    source += baseline.read_text() + '\n' + body + ADAPTER
    ast.parse(source)
    out.write_text(source)

if __name__ == '__main__':
    main()
