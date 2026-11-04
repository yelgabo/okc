#!/bin/bash
# Spill check of cl_phase_a.cuh at several warpgroup register splits (copies the header into .build/cl11).
cd /root/okc/.build/cl11
cp /root/okc/muse_moe_w4a16/cl/cl_math.cuh /root/okc/muse_moe_w4a16/cl/cl_schedule.h .
for v in "40 232" "56 224" "72 216"; do
  set -- $v
  sed "s/REG_LOW = 40, REG_HIGH = 232/REG_LOW = $1, REG_HIGH = $2/" /root/okc/muse_moe_w4a16/cl/cl_phase_a.cuh > cl_phase_a.cuh
  for d in "" "-DCLA_PROF=1"; do
    echo "$1/$2 [$d] $(/usr/local/cuda-13.0/bin/nvcc -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo $d -I. -cubin -o y.cubin inst.cu -Xptxas -v 2>&1 | grep spill)"
  done
done
rm -f cl_phase_a.cuh cl_math.cuh cl_schedule.h
