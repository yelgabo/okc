"""One paper/source feasibility check; no candidate generation or GPU calls."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import torch


def audit(root):
    paths = [root/'.build/today/kel/kernels/muse_decode_layer.py',
             root/'muse_decode_layer/cand/g1b4_stage_release.py']
    pins = ['1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a',
            'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab']
    records = []
    for path, pin in zip(paths, pins):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pin
        source = raw.decode().replace('\r\n', '\n')
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == "_MA_ALT_SRCS['moe_helper0']")
        helper = ast.literal_eval(node.value)
        terms = [line.strip() for line in helper.splitlines()
                 if ('const float gt =' in line or 'const float v0f =' in line)
                 and '(h ? sm.' in line]
        assert len(terms) == 3, terms
        assert all('* gsc' in line for line in terms)
        records.append(dict(source_sha256=pin, helper_sha256=hashlib.sha256(helper.encode()).hexdigest(),
                            source_assignment_line=node.lineno, selected_scaled_sites=terms,
                            helper_scale_definitions=[x.strip() for x in helper.splitlines()
                                if 'const float gsc =' in x and 'TWO62' in x]))
    assert all('fork_bf16' not in x for x in records[0]['selected_scaled_sites'])
    assert all('fork_bf16' in x for x in records[1]['selected_scaled_sites'])
    # Exact binary32 inputs, positive finite scale; CPU BF16 conversion is real Torch.
    v = torch.tensor(1.00390625, dtype=torch.float32)
    scale = torch.tensor(1.0078125, dtype=torch.float32)
    before = (v * scale).bfloat16().float()
    after = (v.bfloat16().float() * scale).bfloat16().float()
    assert before.item() == 1.015625 and after.item() == 1.0078125
    reference = root/'muse_decode_layer/reference.py'
    text = reference.read_text(encoding='utf8')
    for name in ('gs13', 'gs2'):
        assert name+' = '+name+' * torch.exp(torch.randn(' in text
    return dict(status='CLOSED_NO_EXACT_DEPENDENCY_REMOVAL', sources=records,
                reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                counterexample=dict(v=v.item(), scale=scale.item(),
                    round_after_scale=before.item(), round_before_scale_then_round=after.item()),
                scale_product_preexists_G0=True, added_required_edge='BF16 RNE after scale',
                instruction_evidence='aafc32d short-pair-rne selected Tiny/Tiny2 compiler archives',
                actual_exposed_time_us=None, speedup_forecast=None,
                candidate_generated=False, GPU=False, official_calls=False)


if __name__ == '__main__':
    result = audit(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, indent=2))
