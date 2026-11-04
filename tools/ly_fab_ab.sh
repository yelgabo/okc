#!/bin/bash
# Blueprint variants (tools/dl_ly_fab_patch.py) against ly_micro2 on the box, one bx_box_run.py process per run under
# gpu.lock (--lock releases it only while compiling), --runs 60, PDL pinned on at T = 1, 8, 16, 32 in every arm.
# Rounds of: control, then each variant; a final control; the control repeats give the control-vs-control spread.
#   ssh okc-h200 'cd /root/okc && TAG=b2pre VARS="ly_fab_b2pre" ROUNDS="a b" LOADS="decode 1,decode 16,decode 32,verify 1" \
#     nohup tools/ly_fab_ab.sh > .build/box/lyf_b2pre.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
C=muse_decode_layer/cand
D=.build/box/lyf${TAG:+_$TAG}
CTL=${CTL:-ly_micro2.py}
mkdir -p "$D"
run() {
  python tools/bx_box_run.py muse_decode_layer "$C/$2" \
    --judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --runs 60 \
    --loads "${LOADS:-decode 1,decode 16,decode 32,verify 1}" \
    --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
    --pin-pdl8 1,8,16,32 --skip-pdl8-gate \
    --out "$D/$1.json" > "$D/$1.log" 2>&1
  echo "$(date -u +%T) $1 $2 rc=$?"
}
for r in ${ROUNDS:-a b}; do
  run "c_$r" "$CTL"
  for v in $VARS; do run "${v}_$r" "$v.py"; done
done
run "c_z" "$CTL"
echo "$(date -u +%T) done"
