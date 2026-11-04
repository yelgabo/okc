#!/bin/bash
# SASS check of cl_phase_a_kernel<8> (default build): whole-kernel counts of fences and barriers, and the opcode mix
# of the one-tile and two-tile consumer loops (between the consumer's try_wait sites).
cd /root/okc/.build/cl11
printf '#include "cl_phase_a.cuh"\ntemplate __global__ void cla::cl_phase_a_kernel<8>(cla::PhaseAParams);\n' > inst.cu
/usr/local/cuda-13.0/bin/nvcc -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo -I/root/okc/muse_moe_w4a16/cl \
  -cubin -o final.cubin inst.cu -Xptxas -v 2>&1 | grep -E "Used|spill"
/usr/local/cuda-13.0/bin/cuobjdump -sass final.cubin > final.sass
echo "sha256 cl_phase_a.cuh $(sha256sum /root/okc/muse_moe_w4a16/cl/cl_phase_a.cuh | cut -c1-64)"
echo "whole kernel: instructions $(grep -cE '^ +/\*[0-9a-f]{4}\*/' final.sass)"
for op in MEMBAR ERRBAR CGAERRBAR "BAR.SYNC" "BAR.ARV" RED ATOM STG UBLKCP "SYNCS.ARRIVE" "SYNCS.PHASECHK" HMMA USETMAXREG; do
  printf '  %-16s %s\n' "$op" "$(grep -cE "^ +/\*[0-9a-f]{4}\*/ +(@!?U?P[0-9T] +)?$op" final.sass)"
done
echo "lines with MEMBAR, BAR.SYNC, RED, ATOM or STG:"
grep -nE "MEMBAR|BAR\.SYNC|REDG|REDUX|ATOMG|STG|UBLKCP" final.sass | sed 's/ *\/\* 0x[0-9a-f]* \*\///' | cut -c1-110
