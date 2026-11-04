#!/bin/bash
# CL-1.1 final batch: correctness on 60 inputs, then timing (default build, fill and s2 policies), then the
# CLA_PROF timeline. Each python process takes the GPU lock itself.
cd /root/okc && . tools/box_env.sh
P=muse_moe_w4a16/cl/profiles/1.1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
python muse_moe_w4a16/cl/tests/test_phase_a.py --tokens 1,8,32 --inputs 20 --lock /root/okc/.build/box/gpu.lock \
  --json $P/test_phase_a.json > $P/test_phase_a.log 2>&1
grep "SUMMARY\|RESULT" $P/test_phase_a.log
B="python muse_moe_w4a16/cl/cl_phase_a_bench.py --lock /root/okc/.build/box/gpu.lock --replays 20 --inputs 5"
$B --policy fill --configs 8:7,32:7r,32:n,1:n --out $P/bench_final_fill.json > $P/bench_final_fill.log 2>&1
$B --policy s2 --configs 8:7,32:7r --out $P/bench_final_s2.json > $P/bench_final_s2.log 2>&1
$B --policy fill --define=-DCLA_PROF=1 --configs 8:7,32:7r --out $P/bench_final_prof.json > $P/bench_final_prof.log 2>&1
grep "^SUM" $P/bench_final_fill.log $P/bench_final_s2.log $P/bench_final_prof.log
