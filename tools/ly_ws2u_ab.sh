#!/bin/bash
# ly_ws2u against combo2n on the box: decode 1, decode 16, decode 32, verify 1, --runs 60, PDL pinned on at T = 1, 8,
# 16, 32 in both arms, five separate bx_box_run.py processes under gpu.lock: control, candidate, control, candidate,
# control (three control runs give the control-vs-control spread). STEPS="chk" first runs the candidate once at all
# six loads (12 scored cells, --runs 5) for the reference check and G_RELEASE_STARTUP_OK.
#   ssh okc-h200 'cd /root/okc && CAND=ly_ws2u.py nohup tools/ly_ws2u_ab.sh > .build/ly2u/ab.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
mkdir -p .build/ly2u/ab
C=muse_decode_layer/cand
run() {
  python tools/bx_box_run.py muse_decode_layer "$C/$2" \
    --judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --runs "$4" \
    --loads "$3" \
    --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
    --pin-pdl8 1,8,16,32 --skip-pdl8-gate \
    --out ".build/ly2u/ab/$1.json" > ".build/ly2u/ab/$1.log" 2>&1
  echo "$(date -u +%T) $1 $2 rc=$?"
}
declare -A FILE=([c]=rv_k_combo2n.py [w]=${CAND:-ly_ws2u.py})
for step in ${STEPS:-c_a w_a c_b w_b c_c}; do
  if [ "$step" = chk ]; then
    run chk "${FILE[w]}" "decode 1,decode 16,decode 32,verify 1,verify 16,verify 32" 5
  else
    run "ab_$step" "${FILE[${step%%_*}]}" "decode 1,decode 16,decode 32,verify 1" 60
  fi
done
