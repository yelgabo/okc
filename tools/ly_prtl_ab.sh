#!/bin/bash
# A pooler + router candidate against ly_micro2 on the box: all six loads (12 cells), --runs 60, PDL pinned on at
# T = 1, 8, 16, 32 in both arms, five separate bx_box_run.py processes under gpu.lock (--lock): control, candidate,
# control, candidate, control. STEPS="chk ..." first runs the candidate once at all six loads (--runs 5) for the
# reference check and G_RELEASE_STARTUP_OK. Table: python3 tools/ly_prtl_table.py DIR.
#   ssh okc-h200 'cd /root/okc && CAND=ly_prx.py PFX=prx STEPS="chk c_a w_a c_b w_b c_c" \
#     nohup tools/ly_prtl_ab.sh > .build/prtl/prx.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
D=.build/prtl/${PFX:-ab}
mkdir -p "$D"
C=muse_decode_layer/cand
ALL="decode 1,decode 16,decode 32,verify 1,verify 16,verify 32"
run() {
  python tools/bx_box_run.py muse_decode_layer "$C/$2" \
    --judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --runs "$4" \
    --loads "$3" \
    --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
    --pin-pdl8 1,8,16,32 --skip-pdl8-gate \
    --out "$D/$1.json" > "$D/$1.log" 2>&1
  echo "$(date -u +%T) $1 $2 rc=$?"
}
declare -A FILE=([c]=${CTL:-ly_micro2.py} [w]=${CAND:?CAND=file.py})
for step in ${STEPS:-c_a w_a c_b w_b c_c}; do
  if [ "$step" = chk ]; then
    run chk "${FILE[w]}" "$ALL" 5
  else
    run "ab_$step" "${FILE[${step%%_*}]}" "${LOADS:-$ALL}" 60
  fi
done
echo "$(date -u +%T) done"
