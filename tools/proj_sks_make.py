"""Write the stream-K qkv_gate candidate: the C2 control plus sk configs qkv8s / qkv16s / qkv32s.

The new configs keep the incumbents' geometry (24-row blocks, 4 consumer warps, same stages and
stream) and launch SKS_G = 128 CTAs instead of 96. The 96 x tst (row block, k stage) pairs are
laid out row-block-major and CTA i takes stages [i P, (i + 1) P), P = 96 tst / 128 (6 at m = 8,
12 at m = 16 / 32), so each CTA streams 3/4 of a row block and spans at most two. Partials go to
an FP32 scratch with red.add; the last CTA of a row block (per-block counter) converts, stores
through the row map and zeroes the scratch and the counter again (docs/proj-port-guide.md,
"Global split-K join", same ordering argument). The stage-major stream needs no new layout: a
stage of one row block is already one contiguous bulk copy, addressed per stage.

    python3 tools/proj_sks_make.py [out]      (default muse_bf16_proj/experiments/qkv_sks.py)
    python3 tools/proj_sks_make.py --clean out   picks only, no tuner extras (submit form)
"""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import proj_sk_xor_control  # noqa: E402


def sub(src, old, new, count=1):
    n = src.count(old)
    assert n == count, f"expected {count} of {old[:80]!r}, found {n}"
    return src.replace(old, new)


def build(clean=False):
    src = proj_sk_xor_control.restore(ROOT).decode("utf8").replace("\r\n", "\n")

    # ---- Python config ----
    src = sub(src, '    "qkv32": (2, 32, 3, 1, 4, 2, 4),\n',
              '    "qkv32": (2, 32, 3, 1, 4, 2, 4),\n'
              '    # Stream-K twins of the three (SK_SKS below): same geometry, 128 CTAs.\n'
              '    "qkv8s": (1, 8, 3, 1, 4, 4, 4),\n'
              '    "qkv16s": (1, 16, 3, 1, 4, 2, 5),\n'
              '    "qkv32s": (2, 32, 3, 1, 4, 2, 4),\n')
    src = sub(src, "SK_NACC = {nm: 1 | SK_NOVOTE for nm in SK_GEOM}\n",
              "SK_NACC = {nm: 1 | SK_NOVOTE for nm in SK_GEOM}\n"
              "# SkCfg::SKS: stream-K over (row block, k stage), SKS_G CTAs; kept by the raw twins.\n"
              "SK_SKS = 2048\n"
              "for _nm in (\"qkv8s\", \"qkv16s\", \"qkv32s\"):\n"
              "    SK_NACC[_nm] |= SK_SKS\n")
    src = sub(src, 'SK_FULL = {nm for nm in SK_GEOM if nm != "qkv32"}\n',
              'SK_FULL = {nm for nm in SK_GEOM if nm not in ("qkv32", "qkv32s")}\n')
    src = sub(src, "nacc = SK_NACC.get(name, 1) if comp else SK_NACC.get(name, 1) & 15\n",
              "nacc = SK_NACC.get(name, 1) if comp else SK_NACC.get(name, 1) & (15 | SK_SKS)\n")
    if clean:
        raise SystemExit("clean form is written only after a win decides which buckets switch")
    src = sub(src, "SK_EXTRA = {}\n",
              'SK_EXTRA = {("qkv_gate", 8): ["qkv8s"], ("qkv_gate", 16): ["qkv16s"], '
              '("qkv_gate", 32): ["qkv32s"]}\n')

    # ---- device: state and helpers ----
    src = sub(src, """template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL,
          bool FUSE>
struct SkCfg {""", """// Stream-K (SkCfg::SKS) state: FP32 partials [token][stream row] and one arrival counter per
// row block. Zero at module load; the last arriver of a row block reads its rows, writes zeros
// back and wraps the counter (atomicInc), so every launch starts and ends clean and nothing is
// memset in the timed path. SKS launches must not overlap each other (one stream here; a layer
// port with PDL or several streams needs per-call state).
constexpr int SKS_G = 128;
constexpr int SKS_CAP = 32 * 2304;
constexpr int SKS_MAXRB = 128;
__device__ __align__(16) float sks_acc[SKS_CAP];
__device__ unsigned int sks_cnt[SKS_MAXRB];

__device__ __forceinline__ void sks_red2(float* p, float a, float b) {
    asm volatile("red.relaxed.gpu.global.add.v2.f32 [%0], {%1, %2};" :: "l"(p), "f"(a), "f"(b) : "memory");
}

__device__ __forceinline__ void sks_red1(float* p, float a) {
    asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" :: "l"(p), "f"(a) : "memory");
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL,
          bool FUSE>
struct SkCfg {""")
    src = sub(src, "    static constexpr bool NOVOTE = (NACC & 1024) != 0;\n",
              "    static constexpr bool NOVOTE = (NACC & 1024) != 0;\n"
              "    // SKS (bit 11): SKS_G CTAs split the row-block-major (row block, k stage) sequence\n"
              "    // evenly; partials meet in sks_acc (see above).\n"
              "    static constexpr bool SKS = (NACC & 2048) != 0;\n")
    src = sub(src, '    static_assert(!FULL || TOKR == 16 * MT || TOKR == 8, "FULL: bad TOKR");\n',
              '    static_assert(!FULL || TOKR == 16 * MT || TOKR == 8, "FULL: bad TOKR");\n'
              '    static_assert(!SKS || (CL == 1 && !FUSE && !ESCG && !HL), "stream-K: plain single-CTA configs");\n')

    # ---- device: work assignment ----
    src = sub(src, "    uint32_t* zrow_p = reinterpret_cast<uint32_t*>(tab + tab_ints);\n",
              "    uint32_t* zrow_p = reinterpret_cast<uint32_t*>(tab + tab_ints * (C::SKS ? 2 : 1));\n")
    src = sub(src, """    const int rb = CL > 1 ? blockIdx.x / CL : blockIdx.x;
    const int crank = CL > 1 ? blockIdx.x % CL : 0;
    const int nrb = CL > 1 ? gridDim.x / CL : gridDim.x;
    const int tst = K / KST;
    const int gs0 = (crank * tst) / CL;
    const int nst = ((crank + 1) * tst) / CL - gs0;
    const uint8_t* wsrc = ws + (size_t)rb * WBY;
    const size_t wstride = (size_t)nrb * WBY;
""", """    const int tst = K / KST;
    // SKS: this CTA owns global stages [g0, g0 + sks_p) of the row-block-major sequence (row
    // block g / tst, k stage g % tst); rb is its first row block, gs0 the k stage it starts at,
    // and its stages cover sks_nseg (1 or 2) row blocks.
    const int sks_p = C::SKS ? (N / R) * tst / (int)gridDim.x : 1;
    const int sks_g0 = C::SKS ? (int)blockIdx.x * sks_p : 0;
    const int rb = C::SKS ? sks_g0 / tst : (CL > 1 ? blockIdx.x / CL : blockIdx.x);
    const int crank = CL > 1 ? blockIdx.x % CL : 0;
    const int nrb = C::SKS ? N / R : (CL > 1 ? gridDim.x / CL : gridDim.x);
    const int gs0 = C::SKS ? sks_g0 % tst : (crank * tst) / CL;
    const int nst = C::SKS ? sks_p : ((crank + 1) * tst) / CL - gs0;
    const int sks_nseg = C::SKS ? (sks_g0 + sks_p - 1) / tst - rb + 1 : 1;
    const uint8_t* wsrc = ws + (size_t)rb * WBY;
    const size_t wstride = (size_t)nrb * WBY;
    auto wstage = [&](int st) -> const uint8_t* {
        if constexpr (C::SKS) {
            const int gq = gs0 + st;
            return ws + ((size_t)(gq % tst) * nrb + rb + gq / tst) * WBY;
        }
        return wsrc + (size_t)(gs0 + st) * wstride;
    };
    auto kstage = [&](int st) { return C::SKS ? (gs0 + st) % tst : gs0 + st; };
""")
    src = sub(src, "wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);",
              "wstage(st), WBY, full0 + 8 * slot);", count=2)
    src = sub(src, "        tma_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);\n",
              "        tma_3d(s0, &xmap, full0 + 8 * slot, 0, 0, kstage(st) * UPS);\n")
    src = sub(src, "(rq * spp + (st - nst)) * UPS : (gs0 + st) * UPS;",
              "(rq * spp + (st - nst)) * UPS : kstage(st) * UPS;")
    for ind in ("            ", "                "):
        src = sub(src, f"""{ind}mbar_expect_tx(pbar, tab_ints * 4);
{ind}bulk_g2s(smem_u32(tab), ptab + (size_t)rb * tab_ints, tab_ints * 4, pbar);
""", f"""{ind}mbar_expect_tx(pbar, tab_ints * 4 * sks_nseg);
{ind}for (int s = 0; s < sks_nseg; ++s) {{
{ind}    bulk_g2s(smem_u32(tab + s * tab_ints), ptab + (size_t)(rb + s) * tab_ints, tab_ints * 4, pbar);
{ind}}}
""")

    # ---- device: the join, shared by consumer and escape warps ----
    src = sub(src, "    if (warp == NW) {\n        if (lane == 0) {\n            bool dead = false;\n",
              """    // SKS join: every consumer and escape thread fences its reds, thread 0 counts this CTA in
    // for each row block it touched, and the CTA that completes a block's count reads it from
    // L2, stores it through the row map and zeroes it (threadFenceReduction ordering, as the wd
    // GRED join in docs/proj-port-guide.md).
    auto sks_finish = [&]() {
        if constexpr (C::SKS) {
            constexpr int JN = NC + NESC * 32;
            const int jt = tid < NC ? tid : tid - 32;
            __threadfence();
            named_bar(2, JN);
            if (tid == 0) {
                for (int s = 0; s < sks_nseg; ++s) {
                    const int b = rb + s;
                    const unsigned lim = (unsigned)((b * tst + tst - 1) / sks_p - (b * tst) / sks_p);
                    last_flag[s] = atomicInc(&sks_cnt[b], lim) == lim;
                }
            }
            named_bar(2, JN);
            constexpr int R4 = R / 4;
            for (int s = 0; s < sks_nseg; ++s) {
                if (!last_flag[s]) {
                    continue;
                }
                __threadfence();
                const int b = rb + s;
                const int32_t* stab = tab + s * tab_ints;
                for (int idx = jt; idx < R4 * M; idx += JN) {
                    const int tt = idx / R4, row = (idx - tt * R4) * 4;
                    float4* a = reinterpret_cast<float4*>(sks_acc + (size_t)tt * N + b * R + row);
                    const float4 v = __ldcg(a);
                    __stcg(a, make_float4(0.f, 0.f, 0.f, 0.f));
                    const int orow = COMP ? stab[2 * pmax + row] : b * R + row;
                    if ((unsigned)orow >= (unsigned)N) {
                        continue;
                    }
                    OutT* o = out + (size_t)tt * N + orow;
                    if constexpr (sizeof(OutT) == 4) {
                        *reinterpret_cast<float4*>(o) = v;
                    } else {
                        const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
                        uint2 pk;
                        pk.x = *reinterpret_cast<const uint32_t*>(&lo);
                        pk.y = *reinterpret_cast<const uint32_t*>(&hi);
                        *reinterpret_cast<uint2*>(o) = pk;
                    }
                }
            }
        }
    };

    if (warp == NW) {
        if (lane == 0) {
            bool dead = false;
""")

    # ---- device: escape warps ----
    src = sub(src, "            int pe = 1;\n            for (int st = 0; st < nst_tot; ++st) {\n",
              """            if constexpr (C::SKS) {
                // Corrections go straight into sks_acc, per row block of the stage, over exactly
                // this CTA's k range of that block (entries below the first stage are skipped).
                int seg = -1, pe = 1, scnt = 0;
                const int32_t* stab = tab;
                for (int st = 0; st < nst; ++st) {
                    const int slot = st % NSTAGE;
                    const int gq = gs0 + st, sg = gq / tst;
                    const int kbeg = (gq % tst) * KST, kend = kbeg + KST;
                    if (!__all_sync(0xffffffffu, mbar_wait<true>(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 4))) {
                        dead = true;
                        break;
                    }
                    if (sg != seg) {
                        seg = sg;
                        stab = tab + sg * tab_ints;
                        scnt = (dead || C::ESCOFF) ? 0 : stab[0];
                        pe = 1;
                        while (pe <= scnt && (stab[2 * pe] & 0xFFFF) < kbeg) {
                            ++pe;
                        }
                    }
                    if (st % NESC != ew) {
                        while (pe <= scnt && (stab[2 * pe] & 0xFFFF) < kend) {
                            ++pe;
                        }
                    } else {
                        const uint32_t xs = sbase + slot * SST;
                        float* accp = sks_acc + (size_t)(rb + sg) * R;
                        while (pe <= scnt) {
                            const int e0 = stab[2 * pe];
                            const int kk = e0 & 0xFFFF;
                            if (kk >= kend) {
                                break;
                            }
                            if (lane < M) {
                                const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
                                const uint32_t xa = xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * M + lane) & 7)) << 4)
                                                    + (kc & 7) * 2;
                                sks_red1(accp + (size_t)lane * N + (e0 >> 16),
                                         __int_as_float(stab[2 * pe + 1]) * lds_bf16(xa));
                            }
                            ++pe;
                        }
                    }
                    fence_proxy_async();
                    __syncwarp();
                    if (lane == 0) {
                        mbar_arrive(empty0 + 8 * slot);
                    }
                }
                __syncwarp();
                sks_finish();
                return;
            }
            int pe = 1;
            for (int st = 0; st < nst_tot; ++st) {
""")

    # ---- device: consumers ----
    src = sub(src, """        if constexpr (C::LATE_ARRIVE) {
            fence_proxy_async();
            __syncwarp();
            if (lane == 0) {
                mbar_arrive(empty0 + 8 * slot);
            }
        }
    }

    float racc[2][MT][4];
""", """        if constexpr (C::LATE_ARRIVE) {
            fence_proxy_async();
            __syncwarp();
            if (lane == 0) {
                mbar_arrive(empty0 + 8 * slot);
            }
        }
        if constexpr (C::SKS) {
            // Last stage of a row block in this CTA's range: add this warp's partials for the
            // block to sks_acc and restart the accumulators.
            if (st == nst - 1 || (gs0 + st + 1) % tst == 0) {
                float* accp = sks_acc + (size_t)(rb + (gs0 + st) / tst) * R;
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        const int row = (wr * NT + nt) * 8 + 2 * t4;
                        const int tok = mt * 16 + g;
                        float v[4];
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            v[e] = acc[0][mt][nt][e];
#pragma unroll
                            for (int c = 1; c < NA; ++c) {
                                v[e] += acc[c][mt][nt][e];
                            }
#pragma unroll
                            for (int c = 0; c < NA; ++c) {
                                acc[c][mt][nt][e] = 0.f;
                            }
                        }
                        if (tok < M) {
                            sks_red2(accp + (size_t)tok * N + row, v[0], v[1]);
                        }
                        if (tok + 8 < M) {
                            sks_red2(accp + (size_t)(tok + 8) * N + row, v[2], v[3]);
                        }
                    }
                }
            }
        }
    }

    if constexpr (C::SKS) {
        sks_finish();
        return;
    }

    float racc[2][MT][4];
""")

    # ---- host ----
    src = sub(src, "    const int tab_bytes = COMP ? (pmax * 2 + C::R) * 4 : 0;\n",
              "    const int tab_bytes = COMP ? (pmax * 2 + C::R) * 4 * (C::SKS ? 2 : 1) : 0;\n"
              "    const int grid = C::SKS ? SKS_G : N / C::R;\n")
    src = sub(src, '    TORCH_CHECK(smem <= 227 * 1024, "sk: shared memory over budget");\n',
              '    TORCH_CHECK(smem <= 227 * 1024, "sk: shared memory over budget");\n'
              '    if constexpr (C::SKS) {\n'
              '        const int tst = K / (C::UPS * 64), tot = N / C::R * tst;\n'
              '        TORCH_CHECK(tot % SKS_G == 0 && tot / SKS_G <= tst && N / C::R <= SKS_MAXRB\n'
              '                    && (size_t)TOKR * N <= (size_t)SKS_CAP, "sks: shape does not split");\n'
              '    }\n')
    src = sub(src, "            lc.gridDim = dim3(N / C::R);\n", "            lc.gridDim = dim3(grid);\n")
    src = sub(src, "        kern<<<N / C::R, C::THREADS, smem, stream>>>(",
              "        kern<<<grid, C::THREADS, smem, stream>>>(")
    return src


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", nargs="?", type=Path, default=ROOT / "muse_bf16_proj/experiments/qkv_sks.py")
    p.add_argument("--clean", action="store_true")
    a = p.parse_args()
    text = build(a.clean)
    a.out.write_text(text)
    print(f"wrote {a.out} SHA256 {hashlib.sha256(text.encode()).hexdigest()}")
