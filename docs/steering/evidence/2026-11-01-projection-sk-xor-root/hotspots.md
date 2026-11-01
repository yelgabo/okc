### muse_bf16_proj: job 9eba96894ee1402e, 1.528x (1.505-1.551)

SGLang kernels from the baseline-as-submission run in `muse_bf16_proj/baseline_profile/status.bde8b4c49c4a46bd.json`.

| shape | weight | SGLang us | ours us | floor us | SGLang/floor | ours/floor | our top kernels | SGLang top kernels |
|---|---|---|---|---|---|---|---|---|
| decode 1 layer=qkv_gate m=1 | 62 | 14.3 | 9.1 | 7.9 | 1.8x | 1.2x | cgemv40_kernel<256, 2> 8.0us 100% | nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 9.3us 85%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 1.7us 15% |
| decode 1 layer=o m=1 | 62 | 6.8 | 5.0 | 3.5 | 1.9x | 1.4x | cgemv_kernel<64, 1> 5.0us 100% | nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT 5.1us 100% |
| decode 1 layer=pooler m=1 | 62 | 21.2 | 13.5 | 14.0 | 1.5x | 1.0x | rd_kernel<1, 2, 16, 8, 1> 13.5us 100% | nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 18.6us 92%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 1.6us 8% |
| decode 1 layer=unpooler m=1 | 62 | 7.1 | 3.7 | 1.7 | 4.1x | 2.1x | cgemv40_kernel<128, 2> 2.4us 100% | nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.0us 72%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 1.5us 28% |
| decode 1 layer=router m=1 | 62 | 10.5 | 3.8 | 1.8 | 6.0x | 2.2x | rgemv_kernel 2.8us 100% | dot_kernel<float, 128, 0, cublasDotParams<cublas 4.0us 53%<br>at::native::unrolled_elementwise_kernel<at::nati 2.0us 26%<br>reduce_1Block_kernel<float, 128, 7, cublasGemvTe 1.6us 21% |
| decode 16 layer=qkv_gate m=16 | 62 | 13.7 | 9.7 | 7.9 | 1.7x | 1.2x | sk_kernel<1, 16, 3, 1, 4, 2, ... 9.1us 100% | nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT 8.1us 82%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 1.8us 18% |
| decode 16 layer=o m=16 | 62 | 6.7 | 5.4 | 3.6 | 1.9x | 1.5x | sk_kernel<1, 16, 2, 2, 2, 2, ... 4.4us 100% | nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 4.6us 100% |
| decode 16 layer=pooler m=16 | 62 | 21.0 | 13.8 | 14.1 | 1.5x | 1.0x | sk_kernel<1, 16, 2, 2, 2, 4, ... 14.2us 100% | nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT 19.3us 92%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 1.8us 8% |
| decode 16 layer=unpooler m=16 | 62 | 18.4 | 13.6 | 14.1 | 1.3x | 1.0x | sk_kernel<1, 16, 2, 2, 2, 2, ... 13.4us 100% | nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 18.5us 100% |
| decode 16 layer=router m=16 | 62 | 14.7 | 4.4 | 1.8 | 8.1x | 2.5x | rt_kernel<2, 16, 8, 8, 3> 3.9us 100% | nvjet_sm90_tss_64x16_64x16_4x1_v_bz_splitK_NNT 8.1us 63%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 3.7us 29%<br>at::native::vectorized_elementwise_kernel<4, at: 1.0us 8% |
| decode 32 layer=qkv_gate m=32 | 62 | 13.9 | 10.6 | 8.0 | 1.7x | 1.3x | sk_kernel<2, 32, 3, 1, 4, 2, ... 10.2us 100% | nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT 8.6us 82%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 1.9us 18% |
| decode 32 layer=o m=32 | 62 | 6.7 | 5.9 | 3.6 | 1.8x | 1.6x | sk_kernel<2, 32, 2, 2, 2, 2, ... 5.1us 100% | nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 5.1us 100% |
| decode 32 layer=pooler m=32 | 62 | 22.0 | 15.1 | 14.1 | 1.6x | 1.1x | sk_kernel<2, 32, 2, 2, 2, 4, ... 14.9us 100% | nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT 20.1us 91%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 1.9us 9% |
| decode 32 layer=unpooler m=32 | 62 | 19.1 | 14.7 | 14.1 | 1.3x | 1.0x | sk_kernel<2, 32, 2, 2, 2, 2, ... 15.1us 100% | nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 19.5us 100% |
| decode 32 layer=router m=32 | 62 | 15.5 | 5.5 | 1.9 | 8.3x | 2.9x | rt_kernel<4, 16, 8, 4, 3> 4.8us 100% | nvjet_sm90_tss_32x64_64x16_4x1_v_bz_splitK_NNN 8.2us 64%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 3.6us 28%<br>at::native::vectorized_elementwise_kernel<4, at: 1.1us 8% |
| verify 1 layer=qkv_gate m=8 | 62 | 14.3 | 9.8 | 7.9 | 1.8x | 1.2x | sk_kernel<1, 8, 3, 1, 4, 4, 4... 7.6us 100% | nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 9.3us 85%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 1.7us 15% |
| verify 1 layer=o m=8 | 62 | 6.6 | 5.5 | 3.5 | 1.9x | 1.6x | rd_kernel<1, 1, 8, 2, 1> 4.9us 100% | nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT 4.3us 100% |
| verify 1 layer=pooler m=8 | 62 | 21.3 | 13.4 | 14.0 | 1.5x | 1.0x | rd_kernel<1, 2, 16, 8, 1> 13.7us 100% | nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 18.9us 92%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 1.7us 8% |
| verify 1 layer=unpooler m=8 | 62 | 7.1 | 4.5 | 1.8 | 4.0x | 2.6x | sk_kernel<1, 8, 1, 1, 8, 2, 4... 3.6us 100% | nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.1us 70%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 1.7us 30% |
| verify 1 layer=router m=8 | 62 | 26.6 | 4.5 | 1.8 | 15.0x | 2.5x | rt_kernel<1, 8, 8, 4, 3> 3.5us 100% | sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32 9.5us 47%<br>sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32 8.4us 42%<br>at::native::unrolled_elementwise_kernel<at::nati 2.2us 11% |
| verify 16 layer=qkv_gate m=128 | 62 | 16.9 | 14.7 | 8.4 | 2.0x | 1.7x | wd_kernel<1, 128, 3, 1, false... 13.9us 100% | nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN 12.4us 81%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 2.9us 19% |
| verify 16 layer=o m=128 | 62 | 7.3 | 6.8 | 4.0 | 1.8x | 1.7x | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_coopB_NNN 5.5us 100% | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 5.5us 100% |
| verify 16 layer=pooler m=128 | 62 | 25.2 | 20.0 | 14.6 | 1.7x | 1.4x | nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN 20.4us 100% | nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT 25.1us 100% |
| verify 16 layer=unpooler m=128 | 62 | 20.2 | 19.0 | 14.6 | 1.4x | 1.3x | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNN 18.7us 100% | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 21.5us 100% |
| verify 16 layer=router m=128 | 62 | 16.3 | 6.6 | 2.2 | 7.4x | 3.0x | wd_kernel<1, 64, 8, 1, true, ... 6.4us 100% | nvjet_sm90_tss_64x32_64x16_2x4_h_bz_splitK_NNT 9.0us 64%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 3.9us 28%<br>at::native::vectorized_elementwise_kernel<4, at: 1.1us 8% |
| verify 32 layer=qkv_gate m=256 | 62 | 19.2 | 16.9 | 9.8 | 2.0x | 1.7x | nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN 20.4us 100% | nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN 19.6us 100% |
| verify 32 layer=o m=256 | 62 | 8.7 | 8.3 | 4.5 | 2.0x | 1.9x | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_coopA_NNN 8.1us 100% | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 8.7us 100% |
| verify 32 layer=pooler m=256 | 62 | 31.1 | 24.4 | 17.4 | 1.8x | 1.4x | nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN 24.4us 100% | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 29.3us 100% |
| verify 32 layer=unpooler m=256 | 62 | 24.5 | 23.4 | 17.4 | 1.4x | 1.3x | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_coopA_NNN 23.5us 100% | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 25.4us 100% |
| verify 32 layer=router m=256 | 62 | 17.4 | 9.3 | 2.7 | 6.5x | 3.5x | wd_kernel<1, 128, 8, 1, true,... 8.9us 100% | nvjet_sm90_tss_64x32_64x16_1x2_h_bz_splitK_NNT 11.1us 72%<br>cublasLt::splitKreduce_kernel<32, 16, int, float 3.3us 21%<br>at::native::vectorized_elementwise_kernel<4, at: 1.1us 7% |
