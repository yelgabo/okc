"Append one prepared-state-local short serial producer to exact public607."
import argparse
import ast
import hashlib
import json
from pathlib import Path

from prod_size_compile_g import bindings

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py'
PIN = 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
EVIDENCE = ROOT / 'docs/steering/evidence/2026-11-02-worker-short-serial-v2'
APPENDIX = '''

_short_serial_parent_selector = _pool_router_fp32
_short_serial_parent_prepare = prepare

def _pool_router_fp32(h, hr, T, state, ppick):
    if (state is not None and state.get("_short_serial_producer") is True
            and GLE_FRT and T in (1, 8) and hr is not None
            and ppick in ("bfp", "bfpf") and state.get("bfp") is not None
            and state.get("router_fp32") is not None
            and h.dtype == torch.bfloat16 and hr.dtype == torch.float32
            and state["router_fp32"].dtype == torch.float32
            and hr.is_contiguous() and h.is_contiguous()):
        return _gle_prior_pool_router_fp32(h, hr, T, state, ppick)
    return _short_serial_parent_selector(h, hr, T, state, ppick)

def prepare(weights):
    state = _short_serial_parent_prepare(weights)
    state["_short_serial_producer"] = True
    return state
'''


def sha(payload):
    return hashlib.sha256(payload.encode() if isinstance(payload, str) else payload).hexdigest()


def build(original):
    assert sha(original) == PIN and len(original.encode()) == 869076
    assert '\r' not in original and original.endswith('\n')
    assert '_short_serial_producer' not in original
    source = original + APPENDIX
    assert len(source.encode()) < 1000000
    ast.parse(source)
    return source


def validate(source, original):
    assert source == build(original), 'not the one exact append-only treatment'
    before, after = bindings(original), bindings(source)
    for name in ('MOE_CUDA_SRC', 'MOE_CPP_SRC', '_MA_ALT_SRCS'):
        assert before[name] == after[name], name
    return dict(base_sha256=PIN, base_bytes=869076, source_sha256=sha(source),
                bytes=len(source.encode()), exact_original_prefix=True,
                scope='state-local serial BF16 pooler plus FP32 router, eligible T1/T8 only',
                GPU=False, CUDA_numerical_equivalence='NOT RUN', performance_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=EVIDENCE / 'generated')
    args = parser.parse_args()
    original = BASE.read_bytes().decode()
    source = build(original)
    manifest = validate(source, original)
    from test_g_short_serial_producer import contract_evidence
    manifest['CPU_contracts'] = contract_evidence(source)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'candidate.py').write_bytes(source.encode())
    (args.out / 'generation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
