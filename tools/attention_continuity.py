"""Isolated, nondefault continuous attention source integration. No judge actions."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

from attention_top100_screen import EXPECTED, LAYER, ROOT, literal


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def once(s, old, new):
    if s.count(old) != 1:
        raise ValueError('source anchor missing/ambiguous: ' + old[:80])
    return s.replace(old, new)


def transform(cuda):
    cuda = once(cuda, '#define OKC_CONT 0 ', '#define OKC_CONT 1 ')
    cuda = once(cuda, 'if (L == 1 && !CONT && !(OKC_NOXCH1 && !M_CL))',
                'if (L == 1 && !(OKC_NOXCH1 && !M_CL))')
    cuda = once(cuda,
        'float* xs = reinterpret_cast<float*>(S.k) + ((warp & 1) * 32 + lane) * 18;',
        'float* xs = (CONT ? reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(&S) + sizeof(Smem<L, FLAT>)) : reinterpret_cast<float*>(S.k)) + ((warp & 1) * 32 + lane) * 18;')
    cuda = once(cuda,
        'const int sm_f = (int)sizeof(Smem<L, true>) + 1024, sm_n = (int)sizeof(Smem<L, false>) + 1024;',
        'const int sm_f = (int)sizeof(Smem<L, true>) + 1024 + ((L == 1 && Cfg<L, true>::WS && OKC_CONT) ? 64 * 18 * sizeof(float) : 0), sm_n = (int)sizeof(Smem<L, false>) + 1024;')
    anchor = '  // segment list in shared memory (keeps the per-segment loop state out of registers)'
    cuda = once(cuda, anchor,
        '  if constexpr (Cfg<L, FLAT>::WS && OKC_CONT) __syncthreads(); // publish initialized full/empty barriers before role split\n' + anchor)
    return cuda


SELECTOR = '''
def _attention_continuity_module(cfg, state):
    if not (cfg["batch"] in (16, 32) and cfg["q_len"] in (1, 8)
            and cfg["window"] == 0 and cfg["page_size"] == 1
            and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
            and cfg["head_dim"] == 64):
        return None
    module = state.get("attention_continuity_module")
    if module is None:
        import os
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major >= 9 else "")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}" + ("a" if major == 9 else "")
        module = load_inline(name=f"muse_att_cont_disjoint_{arch}", cpp_sources=ATT7_CPP_SRC,
            cuda_sources=ATT7_CONT_CUDA_SRC, functions=["run"],
            extra_cuda_cflags=["-DOKC_QDIRECT=0", "-DOKC_PK2=0", "-DOKC_W0ENABLE=1", "-O3", f"-DOKC_SM90={1 if major >= 9 else 0}", "--expt-relaxed-constexpr"], verbose=False)
        state["attention_continuity_module"] = module
    return module
'''


def generate(text):
    if sha(text) != EXPECTED:
        raise ValueError('wrong incumbent source')
    tree = ast.parse(text)
    original = literal(tree, 'ATT7_CUDA_SRC')
    treatment = transform(original)
    # Preserve the complete original literal, deriving the second readable source
    # by the same bounded replacements rather than duplicating 180KB in the upload.
    edits = list(difflib.unified_diff(original.splitlines(True), treatment.splitlines(True)))
    replacements = []
    matcher = difflib.SequenceMatcher(None, original.splitlines(True), treatment.splitlines(True), autojunk=False)
    a, b = original.splitlines(True), treatment.splitlines(True)
    for tag, i, j, k, l in matcher.get_opcodes():
        if tag != 'equal':
            old, new = ''.join(a[i:j]), ''.join(b[k:l])
            if not old:  # use following context for an insertion
                old = a[i]
                new += a[i]
            replacements.append((old, new))
    derive = '\nATT7_CONT_CUDA_SRC = ATT7_CUDA_SRC\n'
    for old, new in replacements:
        derive += f'assert ATT7_CONT_CUDA_SRC.count({old!r}) == 1\n'
        derive += f'ATT7_CONT_CUDA_SRC = ATT7_CONT_CUDA_SRC.replace({old!r}, {new!r})\n'
    derive += SELECTOR + '\n'
    candidate = once(text, '\n_ATT7 = None', derive + '\n_ATT7 = None')
    candidate = once(candidate, 'attention_state=state, prefetch=prefetch, **pdl)',
        '_mod=_attention_continuity_module(cfg, state), attention_state=state, prefetch=prefetch, **pdl)')
    assert len(candidate.encode()) < 1_000_000
    ast.parse(candidate)
    return candidate, original, treatment, ''.join(edits)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    text = (ROOT / LAYER).read_text(encoding='utf8')
    candidate, original, treatment, patch = generate(text)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, content in [('candidate.py', candidate), ('control.cu', original),
                          ('treatment.cu', treatment), ('native.patch', patch)]:
        (args.out / name).write_text(content, encoding='utf8', newline='\n')
    pins = {n: dict(bytes=len(s.encode()), sha256=sha(s)) for n, s in
            [('candidate.py', candidate), ('control.cu', original), ('treatment.cu', treatment)]}
    (args.out / 'manifest.json').write_text(json.dumps(dict(base_sha256=EXPECTED, files=pins,
        scope='source only; no target build, GPU numerical or performance result'), indent=2)+'\n')
    print(json.dumps(pins))
