"""Generate the isolated D32-ATTN-01 full-layer candidate."""
import ast
import argparse
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py'
EXPECTED = 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('missing or ambiguous source anchor: ' + old[:80])
    return text.replace(old, new)


LOOKAHEAD = r'''
  static_assert(STAGES >= 2, "lookahead holds current and next raw stages");
  auto qk_quad = [&](const uint4 kb, float* score) {
    const uint32_t kw[4] = {kb.x, kb.y, kb.z, kb.w};
    float se[4] = {0.f, 0.f, 0.f, 0.f}, so[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      mma_f16(se, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[2 * h]), cvt2h(kw[2 * h]));
      mma_f16(so, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[2 * h + 1]), cvt2h(kw[2 * h + 1]));
    }
    score[0] = se[0] + so[2]; score[1] = se[1] + so[3];
  };
  float scores[4][2];
  if (ntiles > 0) {
    if (ws_acquire(0, tb % STAGES)) csync<WS, NT>();
    const uint8_t* K0 = S.k + (tb % STAGES) * STAGE_BYTES;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int key = 32 * wsub + 8 * nt + g;
      const uint4 kb = lds128(K0 + key * ROWB + (((hk * 4 + tig) ^ (key & 7)) << 4));
      qk_quad(kb, scores[nt]);
    }
  }
  PROF_INIT
  for (int i = 0; i < ntiles; ++i) {
    PROF_MARK(0);
    if (i == 0) DBG(12);
    const uint8_t* V = S.v + ((tb + i) % STAGES) * STAGE_BYTES;
    const int t0 = start + i * TILE;
    const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);
    if (!interior) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = t0 + 32 * wsub + 8 * nt + 2 * tig + e;
          if (pos >= end) scores[nt][e] = -INFINITY;
        }
    }
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) mx = fmaxf(mx, fmaxf(scores[nt][0], scores[nt][1]));
    mx = quad_max(mx);
    const float m_new = fmaxf(m_r[0], mx);
    const bool active = m_new != -INFINITY;
    float alpha = 1.f;
    if (active) {
      alpha = ex2(m_r[0] - m_new);
      m_r[0] = m_new;
    }
    const bool next = i + 1 < ntiles;
    const uint8_t* next_k = nullptr;
    if (next) {
      if (ws_acquire(i + 1, (tb + i + 1) % STAGES)) csync<WS, NT>();
      next_k = S.k + ((tb + i + 1) % STAGES) * STAGE_BYTES;
    }
    float following[4][2];
    float ls = 0.f;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      uint4 kb;
      if (next) {
        const int key = 32 * wsub + 8 * nt + g;
        kb = lds128(next_k + key * ROWB + (((hk * 4 + tig) ^ (key & 7)) << 4));
      }
      if (active) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(scores[nt][e] - m_new);
          scores[nt][e] = pv;
          ls += pv;
        }
      }
      if (next) qk_quad(kb, following[nt]);
    }
    PROF_MARK(2);
    if (active) {
      if constexpr (ONE) {
        l_r[0] = ls;
      } else {
        l_r[0] = l_r[0] * alpha + ls;
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }
      }
#pragma unroll
      for (int ks = 0; ks < (OKC_XHALFPV ? 1 : 2); ++ks) {
        uint32_t b0[8], b1[8];
        load_v_frags(V, 32 * wsub + 16 * ks, hk, g, tig, b0, b1);
        const uint32_t a0 = pack_h2(scores[2 * ks][0], scores[2 * ks][1]);
        const uint32_t a2 = pack_h2(scores[2 * ks + 1][0], scores[2 * ks + 1][1]);
#pragma unroll
        for (int t = 0; t < 8; ++t) mma_f16(o[0][t], a0, 0u, a2, 0u, b0[t], b1[t]);
      }
    }
    __syncwarp();
    if (OKC_XP_ARRALL || lane == 0) mbar_arrive(&S.empty[(tb + i) % STAGES]);
    if (next) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) scores[nt][e] = following[nt][e];
    }
    PROF_MARK(3);
    PROF_TILE;
    if (i == 0) DBG(13);
  }
  PROF_DUMP;
'''


HOST = '''
def _d32_attention_scope(cfg):
    return (17 <= cfg["batch"] <= 32 and cfg["q_len"] == 1
            and cfg["window"] == 0 and cfg["page_size"] == 1
            and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
            and cfg["head_dim"] == 64 and cfg.get("kv_scale", 1.) == 1.
            and cfg["rope_theta"] == 0.
            and 0. < cfg["sm_scale"] < float("inf")
            and 0. < cfg["qk_scale_factor"] < float("inf"))


def _prepare_d32_attention(state):
    _att7()
    major, minor = torch.cuda.get_device_capability()
    sms = _num_sms7(torch.device("cuda"))
    eligible = (major, minor) == (9, 0) and sms == 132
    state["d32_attention_eligible"] = eligible
    if not eligible:
        print("D32_ATTN_PREPARED", "eligible=False", "capability=", major, minor, "sms=", sms, flush=True)
        return
    if state.get("d32_attention_module") is not None:
        return
    import os
    from torch.utils.cpp_extension import load_inline
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    state["d32_attention_module"] = load_inline(
        name="muse_d32_attn_score_lookahead_sm90a_v1", cpp_sources=ATT7_CPP_SRC,
        cuda_sources=ATT7_D32_CUDA_SRC, functions=["run"],
        extra_cuda_cflags=["-DOKC_QDIRECT=0", "-DOKC_PK2=0", "-DOKC_W0ENABLE=1",
                           "-O3", "-DOKC_SM90=1", "--expt-relaxed-constexpr"], verbose=False)
    print("D32_ATTN_PREPARED", "eligible=True", "source_sha256=", ATT7_D32_CUDA_SHA256,
          "capability=", major, minor, "sms=", sms, flush=True)


def _d32_attention_module(cfg, state):
    if not _d32_attention_scope(cfg) or not state.get("d32_attention_eligible", False):
        return None
    module = state.get("d32_attention_module")
    if module is None:
        raise ValueError("D32 attention module must be prepared before capture")
    return module
'''


def native_change(cuda):
    start = cuda.index('  PROF_INIT\n  for (int i = 0; i < ntiles; ++i) {',
                       cuda.index('  uint32_t qa[4][2];'))
    end = cuda.index('  PROF_DUMP;\n', start) + len('  PROF_DUMP;\n')
    original = cuda[start:end]
    replacement = '  if constexpr (FLAT && WS) {\n' + LOOKAHEAD + '\n  } else {\n' + original + '  }\n'
    return original, replacement


def generate(text):
    if sha(text) != EXPECTED:
        raise ValueError('control source identity changed')
    from attention_top100_screen import literal
    original = literal(ast.parse(text), 'ATT7_CUDA_SRC')
    old, new = native_change(original)
    treatment = once(original, old, new)
    derive = '\nATT7_D32_CUDA_SRC = ATT7_CUDA_SRC\n'
    derive += 'assert ATT7_D32_CUDA_SRC.count({!r}) == 1\n'.format(old)
    derive += 'ATT7_D32_CUDA_SRC = ATT7_D32_CUDA_SRC.replace({!r}, {!r})\n'.format(old, new)
    derive += 'ATT7_D32_CUDA_SHA256 = {!r}\n'.format(sha(treatment))
    candidate = once(text, '\n_ATT7 = None', derive + HOST + '\n_ATT7 = None')
    candidate = once(candidate, 'builders = [_att7, bfp__get_ext,',
                     'builders = [lambda: _prepare_d32_attention(st), bfp__get_ext,')
    candidate = once(candidate, 'attention_state=state, prefetch=prefetch, **pdl)',
                     '_mod=_d32_attention_module(cfg, state), attention_state=state, prefetch=prefetch, **pdl)')
    guard = '''    if _mod is not None and _mod is attention_state.get("d32_attention_module"):
        if not _d32_attention_scope(cfg) or not attention_state.get("d32_attention_eligible", False):
            raise ValueError("D32 attention module outside supported metadata family")
        if (n_splits, flat, cs, ncl, nc) != (264, 1, 0, 1, 1) or tma != (1024 << 2):
            raise ValueError("D32 attention launch geometry or flags drift")
'''
    candidate = once(candidate, '    (_mod or _att7()).run(qkv,', guard + '    (_mod or _att7()).run(qkv,')
    ast.parse(candidate)
    if len(candidate.encode()) >= 1_000_000:
        raise ValueError('candidate exceeds plain-source size limit')
    patch = ''.join(difflib.unified_diff(original.splitlines(True), treatment.splitlines(True),
                                       fromfile='control.cu', tofile='treatment.cu'))
    return candidate, original, treatment, patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    candidate, control, treatment, patch = generate(BASE.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    files = {'candidate.py': candidate, 'control.cu': control,
             'treatment.cu': treatment, 'native.patch': patch}
    for name, content in files.items():
        (args.out / name).write_bytes(content.encode())
    pins = {name: {'bytes': len(content.encode()), 'sha256': sha(content)}
            for name, content in files.items()}
    manifest = {'control_file': str(BASE.relative_to(ROOT)), 'control_sha256': EXPECTED,
                'files': pins, 'status': 'SOURCE_ONLY_TARGET_QUALIFICATION_NOT_RUN'}
    (args.out / 'manifest.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode())
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
