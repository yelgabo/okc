#!/bin/bash
# moe-gb lane on the box. PHASE=gate: the candidate's own prepare at decode 16 / 32 (five in-prepare PDL8H guards),
# then chk (reference check at all six loads, --runs 5). PHASE=ab: two paired rounds against CTL (tools/ly_prtl_ab.sh,
# c/w/c/w/c, 60 runs, PDL pinned at 1, 8, 16, 32). Each python run is under timeout: an MFIN hang must not hold
# gpu.lock for the other lanes.
#   ssh okc-h200 'cd /root/okc && CAND=ly_gb4.py TAG=gb4 PHASE=gate setsid nohup tools/ly_gb_ab.sh \
#     > .build/moegb/gb4_gate.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
CAND=${CAND:?}; TAG=${TAG:?}; CTL=${CTL:-ly_stk3.py}
C=(--judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext)
mkdir -p .build/moegb
if [ "$PHASE" = gate ]; then
  echo "$(date -u +%T) gate start"
  timeout 3000 python tools/bx_box_run.py muse_decode_layer "muse_decode_layer/cand/$CAND" --loads "decode 16,decode 32" --runs 20 "${C[@]}" --out ".build/moegb/gate_$TAG.json" > ".build/moegb/gate_$TAG.log" 2>&1
  echo "$(date -u +%T) gate rc=$?"
  grep -E "PDL8H_GUARD|PDL8_AB|Traceback|Error" ".build/moegb/gate_$TAG.log" | head -20
  CTL=$CTL CAND=$CAND PFX=gb_$TAG STEPS="chk" timeout 3000 tools/ly_prtl_ab.sh
else
  CTL=$CTL CAND=$CAND PFX=gb_$TAG STEPS="c_a w_a c_b w_b c_c" timeout 9000 tools/ly_prtl_ab.sh
  CTL=$CTL CAND=$CAND PFX=gb_${TAG}_r2 STEPS="c_a w_a c_b w_b c_c" timeout 9000 tools/ly_prtl_ab.sh
fi
echo "$(date -u +%T) all done"
