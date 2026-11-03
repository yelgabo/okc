### muse_moe_w4a16: job 3e6852d30c5c4535, 2.305x (2.271-2.340)

SGLang kernels from the baseline-as-submission run in `muse_moe_w4a16/baseline_profile/status.eae9a6b7d5dd4809.json`.

| shape | weight | SGLang us | ours us | floor us | SGLang/floor | ours/floor | our top kernels | SGLang top kernels |
|---|---|---|---|---|---|---|---|---|
| decode 1 bs=1 sharding=ep8 | 60 | 76.3 | 26.6 | 11.4 | 6.7x | 2.3x | okc::moe_fused<false> 22.2us 92%<br>okc::ws_finalize 2.0us 8% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 56.9us 62%<br>at::native::unrolled_elementwise_kernel<at::nati 5.8us 6%<br>at::native::reduce_kernel<512, 1, at::native::Re 5.4us 6% |
| decode 16 bs=16 sharding=ep8 | 60 | 187.6 | 95.3 | 52.5 | 3.6x | 1.8x | okc::moe_fused<false> 105.2us 98%<br>okc::ws_finalize 1.9us 2% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 159.0us 77%<br>at::native::reduce_kernel<512, 1, at::native::Re 8.9us 4%<br>at::native::elementwise_kernel<128, 2, at::nativ 8.6us 4% |
| decode 32 bs=32 sharding=ep8 | 60 | 205.6 | 124.1 | 70.8 | 2.9x | 1.8x | okc::moe_fused<false> 160.8us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 178.5us 92%<br>_post_expert_norm_weighted_sum_kernel 4.5us 2%<br>moe_align_block_size_kernel<int> 2.7us 1% |
| verify 1 bs=8 sharding=ep8 | 60 | 147.1 | 68.5 | 37.3 | 3.9x | 1.8x | okc::moe_fused<false> 54.1us 97%<br>okc::ws_finalize 1.7us 3% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 74.9us 64%<br>at::native::reduce_kernel<512, 1, at::native::Re 8.7us 7%<br>at::native::elementwise_kernel<128, 2, at::nativ 6.6us 6% |
| verify 16 bs=128 sharding=ep8 | 60 | 474.0 | 183.1 | 101.3 | 4.7x | 1.8x | okc::moe_fused<true> 180.0us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 420.5us 96%<br>_post_expert_norm_weighted_sum_kernel 5.1us 1%<br>sglang::act_and_mul_kernel<__nv_bfloat16, 3.5us 1% |
| verify 32 bs=256 sharding=ep8 | 60 | 620.8 | 215.3 | 108.9 | 5.7x | 2.0x | okc::moe_fused<true> 250.0us 100% | sglang::device::marlin_moe::Marlin<__nv_bfloat16 583.8us 96%<br>_post_expert_norm_weighted_sum_kernel 5.6us 1%<br>sglang::act_and_mul_kernel<__nv_bfloat16, 5.5us 1% |

