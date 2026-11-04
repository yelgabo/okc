#!/bin/bash
# ly_ws2t probes at decode 16 on the box: in-graph per-kernel times of the candidate and combo2n (--kernels), then the
# MID timeline of the instrumented candidate. TAG names the outputs in .build/box/ly/.
#   ssh okc-h200 'cd /root/okc && TAG=w4 nohup tools/ly_ws2t_probe.sh > /dev/null 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
T=${TAG:-p}
R=(--judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --loads "decode 16"
   --runs 30 --kernels --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json
   --ext-cache /root/.cache/bx_ext --pin-pdl8 1,8,16,32 --skip-pdl8-gate)
python tools/bx_box_run.py muse_decode_layer muse_decode_layer/cand/${CAND:-ly_ws2t.py} "${R[@]}" > .build/box/ly/k_$T.log 2>&1
[ -n "$NOCTL" ] || python tools/bx_box_run.py muse_decode_layer muse_decode_layer/cand/rv_k_combo2n.py "${R[@]}" > .build/box/ly/k_c_$T.log 2>&1
[ -n "$NOTL" ] || python tools/ly_box_moe_tl16.py muse_decode_layer/cand/${TLCAND:-ly_ws2t_tl.py} \
  --judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --loads "decode 16" --inputs 40 \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json \
  --out .build/box/ly/tl16_$T.json > .build/box/ly/tl16_$T.log 2>&1
echo done > .build/box/ly/probe_$T.done
