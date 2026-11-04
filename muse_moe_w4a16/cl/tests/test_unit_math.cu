// CL-0.5: one CTA per unit, computed from cl_math.cuh only (no ring, no mbarriers, weights read from global).
// unit_a: phase A unit (expert e, 16-row block rb of w13) over the expert's tokens -> 8 inter columns per token.
// unit_b: phase B unit (e, rb of w2) over the expert's intermediate rows -> 16 y values and the unit's partial
//         sum of squares per token.
// combine: the final pass (route scale and slot-order weighted add) from given ss and ybuf.
// run_unit_math.py compares all three bitwise against the helper0 debug build.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "../cl_math.cuh"

namespace {

struct UnitArgs {
  const unsigned char* w;      // unit-order copy (wAu or wBu): [expert][rb][36,864 B]
  const float* gs;             // [32]
  const unsigned short* normw; // post_norm_w raw bf16 bits [4096] (phase B)
  const __nv_bfloat16* src;    // x [M, 4096] (phase A) or inter rows [*, 4096] (phase B)
  const int* ue; const int* urb; const int* urow0; const int* untok;  // per test unit
  const int* rows;             // row table: unit u's token i reads src row rows[urow0[u] + i]
  __nv_bfloat16* outA;         // [nunits, 32, 8]
  float* outY;                 // [nunits, 32, 16]
  float* outSq;                // [nunits, 32]
  float* outAcc;               // [nunits, 32, 16]: raw FP32 cross-warp sums, rows g (0..7) and g + 8 (8..15)
  uint32_t zero;
};

template <int NTT, bool PB>
__global__ void __launch_bounds__(256) unit_kernel(UnitArgs a) {
  __shared__ float4 red[NTT][clm::NWU][32];
  const int u = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int e = a.ue[u], rb = a.urb[u], nt = a.untok[u], r0 = a.urow0[u];
  const unsigned char* unit = a.w + ((size_t)e * (PB ? clm::RB_B : clm::RB_A) + rb) * clm::UNIT_BYTES;
  const uint4* base[NTT];
  bool val[NTT];
#pragma unroll
  for (int n = 0; n < NTT; ++n) {
    const int tok = n * 8 + g;
    val[n] = tok < nt;
    const __nv_bfloat16* row = a.src + (val[n] ? (size_t)a.rows[r0 + tok] * clm::HD : 0);
    base[n] = clm::x_frag_base(row, warp, lane);
  }
  auto xf = [&](int n, int c, uint4& lo, uint4& hi) {
    if (val[n]) { lo = base[n][8 * c]; hi = base[n][8 * c + 1]; }
    else { lo = make_uint4(0, 0, 0, 0); hi = lo; }
  };
  float acc[NTT][4];
  clm::unit_warp_mma<NTT>(unit, warp, lane, 16u + a.zero, xf, acc);
#pragma unroll
  for (int n = 0; n < NTT; ++n) red[n][warp][lane] = make_float4(acc[n][0], acc[n][1], acc[n][2], acc[n][3]);
  __syncthreads();
  if (warp >= NTT) return;
  const int n = warp;
#ifdef CL05_NEGCTL
  // negative control: the 8 warp partials in reverse order; the bitwise comparison must report mismatches
  float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int w = clm::NWU - 1; w >= 0; --w) {
    const float4 v = red[n][w][lane];
    sm.x += v.x; sm.y += v.y; sm.z += v.z; sm.w += v.w;
  }
#else
  const float4 sm = clm::sum_warps(&red[n][0][lane], 32);
#endif
  const float gsc = clm::gscale(a.gs[e]);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int tk = n * 8 + 2 * t + h;
    if (tk < nt) {
      a.outAcc[((size_t)u * 32 + tk) * 16 + g] = h ? sm.y : sm.x;
      a.outAcc[((size_t)u * 32 + tk) * 16 + 8 + g] = h ? sm.w : sm.z;
    }
  }
  if (!PB) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = n * 8 + 2 * t + h;
      if (tk < nt) a.outA[((size_t)u * 32 + tk) * 8 + g] = clm::swiglu_bf16(h ? sm.y : sm.x, h ? sm.w : sm.z, gsc);
    }
  } else {
    const int n0 = rb * 16 + g;
    const float nw0 = clm::norm_mul(a.normw[n0]), nw8 = clm::norm_mul(a.normw[n0 + 8]);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = n * 8 + 2 * t + h;
      float v0, v8;
      const float sq = clm::ss_unit(clm::ynorm_pair(h ? sm.y : sm.x, h ? sm.w : sm.z, gsc, nw0, nw8, v0, v8));
      if (tk < nt) {
        a.outY[((size_t)u * 32 + tk) * 16 + g] = v0;
        a.outY[((size_t)u * 32 + tk) * 16 + g + 8] = v8;
        if (g == 0) a.outSq[(size_t)u * 32 + tk] = sq;
      }
    }
  }
}

__global__ void __launch_bounds__(256) combine_kernel(const int* pos, const float* topk_w, const float* ss,
                                                      const float* ybuf, __nv_bfloat16* out) {
  const int tk = blockIdx.x >> 2, col = (blockIdx.x & 3) * 1024 + threadIdx.x * 4;
  int q[8];
  float r[8];
  float4 y[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    q[s] = pos[tk * 8 + s];
    r[s] = q[s] >= 0 ? clm::route_scale(topk_w[tk * 8 + s], ss[q[s]]) : 0.f;
    y[s] = q[s] >= 0 ? *reinterpret_cast<const float4*>(ybuf + (size_t)q[s] * clm::HD + col) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  *reinterpret_cast<uint2*>(out + (size_t)tk * clm::HD + col) = clm::combine4(q, r, y);
}

template <bool PB>
void launch_units(const UnitArgs& a, int nunits, int ntt) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  switch (ntt) {
    case 1: unit_kernel<1, PB><<<nunits, 256, 0, st>>>(a); break;
    case 2: unit_kernel<2, PB><<<nunits, 256, 0, st>>>(a); break;
    case 3: unit_kernel<3, PB><<<nunits, 256, 0, st>>>(a); break;
    case 4: unit_kernel<4, PB><<<nunits, 256, 0, st>>>(a); break;
    default: TORCH_CHECK(false, "ntt must be 1..4");
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "unit_kernel launch");
}

UnitArgs make_args(torch::Tensor w, torch::Tensor gs, torch::Tensor src, torch::Tensor ue, torch::Tensor urb,
                   torch::Tensor urow0, torch::Tensor untok, torch::Tensor rows) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kUInt8 && src.scalar_type() == torch::kBFloat16);
  UnitArgs a{};
  a.w = w.data_ptr<uint8_t>(); a.gs = gs.data_ptr<float>();
  a.src = (const __nv_bfloat16*)src.data_ptr();
  a.ue = ue.data_ptr<int>(); a.urb = urb.data_ptr<int>(); a.urow0 = urow0.data_ptr<int>();
  a.untok = untok.data_ptr<int>(); a.rows = rows.data_ptr<int>();
  a.zero = 0;
  return a;
}

}  // namespace

std::vector<torch::Tensor> unit_a(torch::Tensor w, torch::Tensor gs, torch::Tensor x, torch::Tensor ue,
                                  torch::Tensor urb, torch::Tensor urow0, torch::Tensor untok, torch::Tensor rows,
                                  int64_t ntt) {
  UnitArgs a = make_args(w, gs, x, ue, urb, urow0, untok, rows);
  auto out = torch::zeros({ue.size(0), 32, 8}, x.options());
  auto acc = torch::zeros({ue.size(0), 32, 16}, x.options().dtype(torch::kFloat32));
  a.outA = (__nv_bfloat16*)out.data_ptr(); a.outAcc = acc.data_ptr<float>();
  launch_units<false>(a, (int)ue.size(0), (int)ntt);
  return {out, acc};
}

std::vector<torch::Tensor> unit_b(torch::Tensor w, torch::Tensor gs, torch::Tensor normw, torch::Tensor inter,
                                  torch::Tensor ue, torch::Tensor urb, torch::Tensor urow0, torch::Tensor untok,
                                  torch::Tensor rows, int64_t ntt) {
  UnitArgs a = make_args(w, gs, inter, ue, urb, urow0, untok, rows);
  a.normw = (const unsigned short*)normw.data_ptr();
  auto fo = inter.options().dtype(torch::kFloat32);
  auto y = torch::zeros({ue.size(0), 32, 16}, fo), sq = torch::zeros({ue.size(0), 32}, fo);
  auto acc = torch::zeros({ue.size(0), 32, 16}, fo);
  a.outY = y.data_ptr<float>(); a.outSq = sq.data_ptr<float>(); a.outAcc = acc.data_ptr<float>();
  launch_units<true>(a, (int)ue.size(0), (int)ntt);
  return {y, sq, acc};
}

torch::Tensor combine(torch::Tensor pos, torch::Tensor topk_w, torch::Tensor ss, torch::Tensor ybuf, int64_t M) {
  auto out = torch::zeros({M, 4096}, ybuf.options().dtype(torch::kBFloat16));
  combine_kernel<<<(unsigned)(M * 4), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      pos.data_ptr<int>(), topk_w.data_ptr<float>(), ss.data_ptr<float>(), ybuf.data_ptr<float>(),
      (__nv_bfloat16*)out.data_ptr());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "combine launch");
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unit_a", &unit_a);
  m.def("unit_b", &unit_b);
  m.def("combine", &combine);
}
