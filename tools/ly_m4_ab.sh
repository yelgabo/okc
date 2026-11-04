#!/bin/bash
# moe lane round 4 on the box: gate (the file's own prepare at decode 16 / 32 with the five in-prepare PDL8H guards),
# chk (reference check at all six loads) and two paired rounds against ly_stk2 (tools/ly_prtl_ab.sh, c/w/c/w/c, 60 runs,
# PDL pinned at 1, 8, 16, 32), then the candidate's prologue stamps (tools/ly_box_moe_pro.py) if PRO=file BUILD=mid|b4.
#   ssh okc-h200 'cd /root/okc && CAND=ly_m4rb.py TAG=rb PRO=ly_m4rb_pro16.py BUILD=mid setsid nohup tools/ly_m4_ab.sh \
#     > .build/moe4/rb_run.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
CAND=${CAND:?}; TAG=${TAG:?}; CTL=${CTL:-ly_stk2.py}
C=(--judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext)
mkdir -p .build/moe4
echo "$(date -u +%T) gate start"
python tools/bx_box_run.py muse_decode_layer "muse_decode_layer/cand/$CAND" --loads "decode 16,decode 32" --runs 20 "${C[@]}" --out ".build/moe4/gate_$TAG.json" > ".build/moe4/gate_$TAG.log" 2>&1
echo "$(date -u +%T) gate rc=$?"
grep -E "PDL8H_GUARD|PDL8_AB|Traceback|Error" ".build/moe4/gate_$TAG.log" | head -20
CTL=$CTL CAND=$CAND PFX=m4_$TAG STEPS="chk c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
CTL=$CTL CAND=$CAND PFX=m4_${TAG}_r2 STEPS="c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
if [ -n "$PRO" ]; then
  python tools/ly_box_moe_pro.py "muse_decode_layer/cand/$PRO" --build "$BUILD" --inputs 20 --raw 4 "${C[@]}" --out ".build/moe4/pro_$TAG.json" > ".build/moe4/pro_$TAG.log" 2>&1
  echo "$(date -u +%T) pro rc=$?"
fi
echo "$(date -u +%T) all done"
