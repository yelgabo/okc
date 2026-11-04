#!/bin/bash
# ly_ws2t against combo2n on the box: decode 1, decode 16, decode 32, verify 1, verify 16, --runs 60, PDL pinned on
# at T = 1, 8, 16, 32 in both arms, each run a separate bx_box_run.py process under gpu.lock, control, candidate,
# control, candidate, control (the control twice or more gives the control-vs-control spread).
#   ssh okc-h200 'cd /root/okc && nohup tools/ly_ws2t_ab.sh > .build/box/ly/ab.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
C=muse_decode_layer/cand
run() {
  python tools/bx_box_run.py muse_decode_layer "$C/$2" \
    --judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --runs 60 \
    --loads "decode 1,decode 16,decode 32,verify 1,verify 16" \
    --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
    --pin-pdl8 1,8,16,32 --skip-pdl8-gate \
    --out ".build/box/ly/$1.json" > ".build/box/ly/$1.log" 2>&1
  local rc=$?
  echo "$(date -u +%T) $1 $2 rc=$rc"
}
declare -A FILE=([c]=rv_k_combo2n.py [w]=ly_ws2t.py)
for step in ${STEPS:-c_a w_a c_b w_b c_c}; do
  run "ab_$step" "${FILE[${step%%_*}]}"
done
