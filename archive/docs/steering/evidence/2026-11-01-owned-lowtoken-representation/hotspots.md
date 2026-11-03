### muse_moe_w4a16: job 835253441a214a1c, 2.190x (2.157-2.222)

SGLang kernels from the baseline-as-submission run in `C:/Users/gretc/Documents/GitHub/okc/muse_moe_w4a16/baseline_profile/status.eae9a6b7d5dd4809.json`.

| shape | weight | SGLang us | ours us | floor us | SGLang/floor | ours/floor | our top kernels | SGLang top kernels |
|---|---|---|---|---|---|---|---|---|
| decode 1 bs=1 sharding=ep8 | 60 | 80.3 | 28.7 | 11.4 | 7.0x | 2.5x | sk_8_3_0::moe_fused_kernel<1>... 23.8us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 56.9us 62%<br>at::native::unrolled_elementwise_kernel<at::nati 5.8us 6%<br>at::native::reduce_kernel<512, 1, at::native::Re 5.4us 6% |
| decode 16 bs=16 sharding=ep8 | 60 | 187.2 | 95.4 | 52.5 | 3.6x | 1.8x | sk_8_3_0::moe_fused_kernel<1>... 86.9us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 159.0us 77%<br>at::native::reduce_kernel<512, 1, at::native::Re 8.9us 4%<br>at::native::elementwise_kernel<128, 2, at::nativ 8.6us 4% |
| decode 32 bs=32 sharding=ep8 | 60 | 205.9 | 132.6 | 70.8 | 2.9x | 1.9x | sk_8_3_0::moe_fused_kernel<1>... 144.0us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 178.5us 92%<br>_post_expert_norm_weighted_sum_kernel 4.5us 2%<br>moe_align_block_size_kernel<int> 2.7us 1% |
| verify 1 bs=8 sharding=ep8 | 60 | 142.2 | 66.7 | 37.3 | 3.8x | 1.8x | sk_8_3_0::moe_fused_kernel<1>... 79.9us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 74.9us 64%<br>at::native::reduce_kernel<512, 1, at::native::Re 8.7us 7%<br>at::native::elementwise_kernel<128, 2, at::nativ 6.6us 6% |
| verify 16 bs=128 sharding=ep8 | 60 | 468.3 | 199.3 | 101.3 | 4.6x | 2.0x | moe_main_kernel<47> 179.3us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 420.5us 96%<br>_post_expert_norm_weighted_sum_kernel 5.1us 1%<br>sglang::act_and_mul_kernel<__nv_bfloat16, 3.5us 1% |
| verify 32 bs=256 sharding=ep8 | 60 | 616.6 | 239.5 | 108.9 | 5.7x | 2.2x | moe_main_kernel<47> 230.1us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 583.8us 96%<br>_post_expert_norm_weighted_sum_kernel 5.6us 1%<br>sglang::act_and_mul_kernel<__nv_bfloat16, 5.5us 1% |

