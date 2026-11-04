#!/bin/bash
# CL-1.1 experiment batch: solo slices under the experiment builds, and the s2 policy on the full grid.
cd /root/okc && . tools/box_env.sh
B="python muse_moe_w4a16/cl/cl_phase_a_bench.py --lock /root/okc/.build/box/gpu.lock --inputs 2 --replays 10"
for d in "-DCLA_PROF=1" "-DCLA_PROF=1 -DCLA_EXP=1" "-DCLA_PROF=1 -DCLA_EXP=2"; do
  for solo in first hot; do
    echo "== defines [$d] solo $solo"
    $B --define="$d" --configs 8:7,32:7r --solo $solo 2>&1 | grep "^RUN" | cut -c1-24,40-700
  done
done
echo "== full grid, CLA_PROF, policy s2 and fill"
for pol in s2 fill; do
  $B --define=-DCLA_PROF=1 --policy $pol --configs 8:7,32:7r 2>&1 | grep "^RUN" | cut -c1-24,40-700
done
