"""Build state-local attention arm selection for a private matched instrument."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'docs/steering/decode32/evidence/D32-ATTN-01/candidate.py'
PIN = '5a580c7a0fad1246503ebc66cfbeca698c2b011638070f340480e4560a18d3b5'

SUPPORT = '''
_d32_pair_parent_module = _d32_attention_module

def _d32_pair_bind(state, treatment):
    if type(treatment) is not bool or '_d32_pair_bound' in state:
        raise RuntimeError('invalid or repeated D32 arm binding')
    state['_d32_pair_bound'] = treatment
    state['_d32_pair_arm'] = treatment
    return state

def _d32_attention_module(cfg, state):
    arm = state.get('_d32_pair_arm', True)
    bound = state.get('_d32_pair_bound', True)
    if type(arm) is not bool or type(bound) is not bool or arm is not bound:
        raise RuntimeError('D32 arm changed after binding')
    if not arm:
        return None
    return _d32_pair_parent_module(cfg, state)
'''


def build(text):
    if hashlib.sha256(text.encode()).hexdigest() != PIN:
        raise ValueError('attention candidate identity changed')
    result = text + '\n' + SUPPORT
    ast.parse(result)
    if len(result.encode()) >= 1_000_000:
        raise ValueError('candidate exceeds size limit')
    return result
