#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define SILU_DIV(a, b) __fdividef((a), (b))
#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)

namespace okc_t1 {

@@SELECTED_HELPERS@@

struct Params {
  const unsigned char* weights_a;
  const unsigned char* weights_b;
  const float* scale_a;
  const float* scale_b;
  const __nv_bfloat16* norm;
  const __nv_bfloat16* input;
  const int* ids;
  const float* route_weights;
  __nv_bfloat16* inter;
  float* expert_y;
  float* partials;
  __nv_bfloat16* output;
  int local_start;
  int local_count;
};

__device__ __forceinline__ bool local_route(const Params& params, int slot) {
  const int expert = params.ids[slot];
  return expert >= params.local_start && expert < params.local_start + params.local_count;
}

template <bool DOWN>
__device__ __forceinline__ void tile(const Params& params, float (&reduction)[8][32][4]) {
  const int slot = blockIdx.y;
  if (!local_route(params, slot)) return;
  const int expert = params.ids[slot] - params.local_start;
  const int row_block = blockIdx.x;
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int group = lane >> 2;
  const int token_pair = lane & 3;
  const int row_blocks = DOWN ? 256 : 512;
  const unsigned char* unit = (DOWN ? params.weights_b : params.weights_a)
      + ((size_t)expert * row_blocks + row_block) * 36864;
  const uint4 scales = *reinterpret_cast<const uint4*>(unit + 32768 + warp * 512 + lane * 16);
  const __nv_bfloat16* input = DOWN ? params.inter + (size_t)slot * 4096 : params.input;
  uint4 inputs[16];
#pragma unroll
  for (int chunk = 0; chunk < 8; ++chunk) {
    inputs[2 * chunk] = group == 0
        ? __ldg(reinterpret_cast<const uint4*>(input + warp * 512 + 16 * token_pair + chunk * 64))
        : make_uint4(0, 0, 0, 0);
    inputs[2 * chunk + 1] = group == 0
        ? __ldg(reinterpret_cast<const uint4*>(input + warp * 512 + 16 * token_pair + chunk * 64 + 8))
        : make_uint4(0, 0, 0, 0);
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  float acc2[4] = {0.f, 0.f, 0.f, 0.f};
  float acc3[4] = {0.f, 0.f, 0.f, 0.f};
  float acc4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int chunk = 0; chunk < 8; ++chunk) {
    const uint4 weights = *reinterpret_cast<const uint4*>(unit + (warp * 8 + chunk) * 512 + lane * 16);
    const uint32_t scale_word = chunk < 4 ? (chunk < 2 ? scales.x : scales.y)
                                          : (chunk < 6 ? scales.z : scales.w);
    const uint32_t gate_scale = SCDK(scale_word, (chunk & 1) * 2, 16u);
    const uint32_t up_scale = SCDK(scale_word, (chunk & 1) * 2 + 1, 16u);
    const uint32_t packed[4] = {weights.x, weights.y, weights.z, weights.w};
    const uint32_t values[8] = {inputs[2 * chunk].x, inputs[2 * chunk].y,
        inputs[2 * chunk].z, inputs[2 * chunk].w, inputs[2 * chunk + 1].x,
        inputs[2 * chunk + 1].y, inputs[2 * chunk + 1].z, inputs[2 * chunk + 1].w};
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      uint32_t first, second, third, fourth;
      dq4(packed[step], first, second, third, fourth);
      mma_bf16(step == 0 ? acc : step == 1 ? acc2 : step == 2 ? acc3 : acc4,
          bmul2(first, gate_scale), bmul2(second, up_scale), bmul2(third, gate_scale),
          bmul2(fourth, up_scale), values[2 * step], values[2 * step + 1]);
    }
  }
#pragma unroll
  for (int component = 0; component < 4; ++component)
    acc[component] += acc2[component] + acc3[component] + acc4[component];
  *reinterpret_cast<float4*>(&reduction[warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
  __syncthreads();
  if (thread < 32) {
    float gate_or_first = 0.f;
    float up_or_second = 0.f;
#pragma unroll
    for (int source_warp = 0; source_warp < 8; ++source_warp) {
      gate_or_first += reduction[source_warp][lane][0];
      up_or_second += reduction[source_warp][lane][2];
    }
    const float global_scale = (DOWN ? params.scale_b[expert] : params.scale_a[expert])
        * 4.611686018427388e18f;
    if (!DOWN) {
      if (token_pair == 0) {
        const float gate = fork_bf16(gate_or_first * global_scale);
        const float up = fork_bf16(up_or_second * global_scale);
        params.inter[(size_t)slot * 4096 + row_block * 8 + group]
            = __float2bfloat16(probe_q(SILU_DIV(gate, 1.f + __expf(-gate)) * up));
      }
    } else {
      const int column = row_block * 16 + group;
      const float gamma_first = fork_bf16(1.f + __bfloat162float(params.norm[column]));
      const float gamma_second = fork_bf16(1.f + __bfloat162float(params.norm[column + 8]));
      const float first = fork_bf16(gate_or_first * global_scale) * gamma_first;
      const float second = fork_bf16(up_or_second * global_scale) * gamma_second;
      float square = token_pair == 0 ? first * first + second * second : 0.f;
      if (token_pair == 0) {
        params.expert_y[(size_t)slot * 4096 + column] = first;
        params.expert_y[(size_t)slot * 4096 + column + 8] = second;
      }
      square += __shfl_xor_sync(0xffffffffu, square, 4);
      square += __shfl_xor_sync(0xffffffffu, square, 8);
      square += __shfl_xor_sync(0xffffffffu, square, 16);
      if (lane == 0) params.partials[(size_t)slot * 256 + row_block] = square;
    }
  }
}

__global__ __launch_bounds__(256) void gate_up(Params params) {
  __shared__ float reduction[8][32][4];
  tile<false>(params, reduction);
}

__global__ __launch_bounds__(256) void down(Params params) {
  __shared__ float reduction[8][32][4];
  tile<true>(params, reduction);
}

__global__ __launch_bounds__(256) void norm_combine(Params params) {
  __shared__ float warp_sums[8];
  __shared__ float inverse[8];
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  for (int slot = 0; slot < 8; ++slot) {
    float total = local_route(params, slot) ? params.partials[slot * 256 + thread] : 0.f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      total += __shfl_down_sync(0xffffffffu, total, offset);
    if (lane == 0) warp_sums[warp] = total;
    __syncthreads();
    if (thread == 0) {
      float square_sum = 0.f;
      for (int source_warp = 0; source_warp < 8; ++source_warp) square_sum += warp_sums[source_warp];
      inverse[slot] = local_route(params, slot) ? rsqrtf(square_sum * (1.f / 4096) + 1e-8f) : 0.f;
    }
    __syncthreads();
  }
  for (int column = thread; column < 4096; column += 256) {
    float combined = 0.f;
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
      if (local_route(params, slot)) {
        const float weight = params.route_weights[slot] * inverse[slot];
        combined += weight * params.expert_y[slot * 4096 + column];
      }
    }
    params.output[column] = __float2bfloat16_rn(combined);
  }
}

}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void t1_three_run(std::vector<torch::Tensor> weights, std::vector<torch::Tensor> scratch,
    torch::Tensor input, torch::Tensor ids, torch::Tensor route_weights, torch::Tensor output) {
  TORCH_CHECK(weights.size() >= 12 && scratch.size() == 3, "T1_THREE state ABI");
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type() == torch::kBFloat16
      && input.dim() == 2 && input.size(0) == 1 && input.size(1) == 4096, "T1_THREE input ABI");
  TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type() == torch::kInt32
      && ids.numel() == 8 && ids.device() == input.device(), "T1_THREE route ABI");
  TORCH_CHECK(route_weights.is_cuda() && route_weights.is_contiguous()
      && route_weights.scalar_type() == torch::kFloat32 && route_weights.numel() == 8
      && route_weights.device() == input.device(), "T1_THREE score ABI");
  TORCH_CHECK(output.is_cuda() && output.is_contiguous() && output.scalar_type() == torch::kBFloat16
      && output.sizes() == input.sizes() && output.device() == input.device(), "T1_THREE output ABI");
  const int indices[5] = {2, 5, 7, 10, 11};
  for (int index : indices)
    TORCH_CHECK(weights[index].is_cuda() && weights[index].is_contiguous()
        && weights[index].device() == input.device(), "T1_THREE weight placement");
  TORCH_CHECK(weights[2].scalar_type() == torch::kFloat32 && weights[2].numel() == 32
      && weights[5].scalar_type() == torch::kFloat32 && weights[5].numel() == 32
      && weights[7].scalar_type() == torch::kBFloat16 && weights[7].numel() == 4096
      && weights[10].scalar_type() == torch::kUInt8 && weights[10].numel() == 32LL * 512 * 36864
      && weights[11].scalar_type() == torch::kUInt8 && weights[11].numel() == 32LL * 256 * 36864,
      "T1_THREE immutable packing ABI");
  for (int index = 0; index < 3; ++index)
    TORCH_CHECK(scratch[index].is_cuda() && scratch[index].is_contiguous()
        && scratch[index].device() == input.device(), "T1_THREE scratch placement");
  TORCH_CHECK(scratch[0].scalar_type() == torch::kBFloat16 && scratch[0].numel() == 8 * 4096
      && scratch[1].scalar_type() == torch::kFloat32 && scratch[1].numel() == 8 * 4096
      && scratch[2].scalar_type() == torch::kFloat32 && scratch[2].numel() == 8 * 256,
      "T1_THREE scratch ABI");
  okc_t1::Params params{};
  params.weights_a = (const unsigned char*)weights[10].data_ptr();
  params.weights_b = (const unsigned char*)weights[11].data_ptr();
  params.scale_a = (const float*)weights[2].data_ptr();
  params.scale_b = (const float*)weights[5].data_ptr();
  params.norm = (const __nv_bfloat16*)weights[7].data_ptr();
  params.input = (const __nv_bfloat16*)input.data_ptr();
  params.ids = (const int*)ids.data_ptr();
  params.route_weights = (const float*)route_weights.data_ptr();
  params.output = (__nv_bfloat16*)output.data_ptr();
  params.inter = (__nv_bfloat16*)scratch[0].data_ptr();
  params.expert_y = (float*)scratch[1].data_ptr();
  params.partials = (float*)scratch[2].data_ptr();
  params.local_start = 0;
  params.local_count = 32;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  okc_t1::gate_up<<<dim3(512, 8), 256, 0, stream>>>(params);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "T1_THREE gate launch");
  okc_t1::down<<<dim3(256, 8), 256, 0, stream>>>(params);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "T1_THREE down launch");
  okc_t1::norm_combine<<<1, 256, 0, stream>>>(params);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "T1_THREE norm/combine launch");
}
