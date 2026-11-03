### muse_moe_w4a16: job 56e0c8b5634545e6, 2.332x (2.297-2.367)

SGLang kernels from the baseline-as-submission run in `muse_moe_w4a16/baseline_profile/profiles/baseline.eae9a6b7d5dd4809.json`.

| shape | weight | SGLang us | ours us | floor us | SGLang/floor | ours/floor | our top kernels | SGLang top kernels |
|---|---|---|---|---|---|---|---|---|
| decode 1 bs=1 sharding=ep8 | 60 | 77.3 | 25.1 | 11.4 | 6.8x | 2.2x | okc::moe_fused<false> 23.6us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 56.9us 62%<br>at::native::unrolled_elementwise_kernel<at::nati 5.8us 6%<br>at::native::reduce_kernel<512, 1, at::native::Re 5.4us 6% |
| decode 16 bs=16 sharding=ep8 | 60 | 189.2 | 96.5 | 52.5 | 3.6x | 1.8x | okc::moe_fused<false> 84.6us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 159.0us 77%<br>at::native::reduce_kernel<512, 1, at::native::Re 8.9us 4%<br>at::native::elementwise_kernel<128, 2, at::nativ 8.6us 4% |
| decode 32 bs=32 sharding=ep8 | 60 | 203.8 | 123.0 | 70.8 | 2.9x | 1.7x | okc::moe_fused<false> 122.0us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 178.5us 92%<br>_post_expert_norm_weighted_sum_kernel 4.5us 2%<br>moe_align_block_size_kernel<int> 2.7us 1% |
| verify 1 bs=8 sharding=ep8 | 60 | 142.2 | 64.1 | 37.3 | 3.8x | 1.7x | okc::moe_fused<false> 77.0us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 74.9us 64%<br>at::native::reduce_kernel<512, 1, at::native::Re 8.7us 7%<br>at::native::elementwise_kernel<128, 2, at::nativ 6.6us 6% |
| verify 16 bs=128 sharding=ep8 | 60 | 461.8 | 180.2 | 101.3 | 4.6x | 1.8x | okc::moe_fused<true> 150.3us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 420.5us 96%<br>_post_expert_norm_weighted_sum_kernel 5.1us 1%<br>sglang::act_and_mul_kernel<__nv_bfloat16, 3.5us 1% |
| verify 32 bs=256 sharding=ep8 | 60 | 619.2 | 218.8 | 108.9 | 5.7x | 2.0x | okc::moe_fused<true> 227.3us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 583.8us 96%<br>_post_expert_norm_weighted_sum_kernel 5.6us 1%<br>sglang::act_and_mul_kernel<__nv_bfloat16, 5.5us 1% |
