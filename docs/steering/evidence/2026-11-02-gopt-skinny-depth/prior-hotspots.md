### muse_moe_w4a16: job 75bca9775f6f4b0b, 2.006x (1.976-2.036)

SGLang kernels from the baseline-as-submission run in `muse_moe_w4a16/baseline_profile/profiles/baseline.eae9a6b7d5dd4809.json`.

| shape | weight | SGLang us | ours us | floor us | SGLang/floor | ours/floor | our top kernels | SGLang top kernels |
|---|---|---|---|---|---|---|---|---|
| decode 1 bs=1 sharding=ep8 | 60 | 80.3 | 28.6 | 11.4 | 7.0x | 2.5x | sk_8_3_0::moe_fused_kernel<1>... 24.3us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 56.9us 62%<br>at::native::unrolled_elementwise_kernel<at::nati 5.8us 6%<br>at::native::reduce_kernel<512, 1, at::native::Re 5.4us 6% |
| decode 16 bs=16 sharding=ep8 | 60 | 188.5 | 96.6 | 52.5 | 3.6x | 1.8x | sk_8_3_0::moe_fused_kernel<1>... 72.5us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 159.0us 77%<br>at::native::reduce_kernel<512, 1, at::native::Re 8.9us 4%<br>at::native::elementwise_kernel<128, 2, at::nativ 8.6us 4% |
| decode 32 bs=32 sharding=ep8 | 60 | 212.7 | 137.3 | 70.8 | 3.0x | 1.9x | sk_8_3_0::moe_fused_kernel<1>... 117.6us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 178.5us 92%<br>_post_expert_norm_weighted_sum_kernel 4.5us 2%<br>moe_align_block_size_kernel<int> 2.7us 1% |
| verify 1 bs=8 sharding=ep8 | 60 | 149.4 | 70.8 | 37.3 | 4.0x | 1.9x | sk_8_3_0::moe_fused_kernel<1>... 73.2us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 74.9us 64%<br>at::native::reduce_kernel<512, 1, at::native::Re 8.7us 7%<br>at::native::elementwise_kernel<128, 2, at::nativ 6.6us 6% |
| verify 16 bs=128 sharding=ep8 | 60 | 465.4 | 263.9 | 101.3 | 4.6x | 2.6x | w4_gemm_kernel<1, 1, 8, 2, 8,... 187.4us 62%<br>w4_gemm_kernel<2, 2, 16, 4, 4... 84.3us 28%<br>w4_swiglu_kernel 14.5us 5% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 420.5us 96%<br>_post_expert_norm_weighted_sum_kernel 5.1us 1%<br>sglang::act_and_mul_kernel<__nv_bfloat16, 3.5us 1% |
| verify 32 bs=256 sharding=ep8 | 60 | 615.6 | 298.3 | 108.9 | 5.7x | 2.7x | moe_main_kernel<0> 328.4us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 583.8us 96%<br>_post_expert_norm_weighted_sum_kernel 5.6us 1%<br>sglang::act_and_mul_kernel<__nv_bfloat16, 5.5us 1% |
