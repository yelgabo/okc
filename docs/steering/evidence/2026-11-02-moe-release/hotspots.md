### muse_moe_w4a16: job 64f9998ef238476f, 2.334x (2.299-2.369)

SGLang kernels from the baseline-as-submission run in `muse_moe_w4a16/baseline_profile/status.eae9a6b7d5dd4809.json`.

| shape | weight | SGLang us | ours us | floor us | SGLang/floor | ours/floor | our top kernels | SGLang top kernels |
|---|---|---|---|---|---|---|---|---|
| decode 1 bs=1 sharding=ep8 | 60 | 76.8 | 25.3 | 11.4 | 6.7x | 2.2x | okc::moe_fused<false> 22.9us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 56.9us 62%<br>at::native::unrolled_elementwise_kernel<at::nati 5.8us 6%<br>at::native::reduce_kernel<512, 1, at::native::Re 5.4us 6% |
| decode 16 bs=16 sharding=ep8 | 60 | 188.3 | 95.4 | 52.5 | 3.6x | 1.8x | okc::moe_fused<false> 73.7us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 159.0us 77%<br>at::native::reduce_kernel<512, 1, at::native::Re 8.9us 4%<br>at::native::elementwise_kernel<128, 2, at::nativ 8.6us 4% |
| decode 32 bs=32 sharding=ep8 | 60 | 210.8 | 127.4 | 70.8 | 3.0x | 1.8x | okc::moe_fused<false> 146.5us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 178.5us 92%<br>_post_expert_norm_weighted_sum_kernel 4.5us 2%<br>moe_align_block_size_kernel<int> 2.7us 1% |
| verify 1 bs=8 sharding=ep8 | 60 | 142.8 | 64.6 | 37.3 | 3.8x | 1.7x | okc::moe_fused<false> 71.9us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 74.9us 64%<br>at::native::reduce_kernel<512, 1, at::native::Re 8.7us 7%<br>at::native::elementwise_kernel<128, 2, at::nativ 6.6us 6% |
| verify 16 bs=128 sharding=ep8 | 60 | 460.9 | 179.0 | 101.3 | 4.5x | 1.8x | okc::moe_fused<true> 197.0us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 420.5us 96%<br>_post_expert_norm_weighted_sum_kernel 5.1us 1%<br>sglang::act_and_mul_kernel<__nv_bfloat16, 3.5us 1% |
| verify 32 bs=256 sharding=ep8 | 60 | 630.3 | 220.0 | 108.9 | 5.8x | 2.0x | okc::moe_fused<true> 214.6us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 583.8us 96%<br>_post_expert_norm_weighted_sum_kernel 5.6us 1%<br>sglang::act_and_mul_kernel<__nv_bfloat16, 5.5us 1% |
