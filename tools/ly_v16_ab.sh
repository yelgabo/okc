#!/bin/bash
# layer-d16 lane, round 2, on the box: a gate run of CAND at GATE_LOADS with the file's own in-prepare guards (no pin),
# two paired A/B rounds against CTL at all six loads (tools/ly_prtl_ab.sh, chk step first), then a kernel timeline of
# CAND at TL_LOADS (tools/ly_tl_run.py). Logs under .build/d16r2/ and .build/prtl/<TAG>{,_r2}/.
#   ssh okc-h200 'cd /root/okc && CAND=ly_v16pdl.py TAG=v16pdl nohup bash tools/ly_v16_ab.sh > .build/d16r2/v16pdl_run.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
CAND=${CAND:?CAND=file.py}; TAG=${TAG:?TAG=name}; CTL=${CTL:-ly_stk1.py}
GATE_LOADS=${GATE_LOADS:-verify 16,verify 32}; TL_LOADS=${TL_LOADS:-verify 16,verify 32}
COMMON=(--lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext
        --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json)
mkdir -p .build/d16r2
if [ -z "$NOGATE" ]; then
  echo "$(date -u +%T) gate start"
  python tools/bx_box_run.py muse_decode_layer "muse_decode_layer/cand/$CAND" --loads "$GATE_LOADS" --runs 20 \
    "${COMMON[@]}" --out ".build/d16r2/gate_$TAG.json" > ".build/d16r2/gate_$TAG.log" 2>&1
  echo "$(date -u +%T) gate rc=$?"
fi
[ -n "$GATEONLY" ] && exit 0
CTL=$CTL CAND=$CAND PFX=$TAG STEPS="chk c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
CTL=$CTL CAND=$CAND PFX=${TAG}_r2 STEPS="c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
if [ -z "$NOTL" ]; then
  LY_TRACE_DIR=".build/d16r2/tl_$TAG" python tools/ly_tl_run.py muse_decode_layer "muse_decode_layer/cand/$CAND" \
    --loads "$TL_LOADS" --runs 20 --kernels --pin-pdl8 1,8,16,32 --skip-pdl8-gate "${COMMON[@]}" \
    --out ".build/d16r2/tl_$TAG.json" > ".build/d16r2/tl_$TAG.log" 2>&1
fi
echo "$(date -u +%T) all done"
