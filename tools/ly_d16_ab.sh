#!/bin/bash
# layer-d16 lane on the box: a gate run of CAND at decode 16 / 32 with the file's own in-prepare PDL8 guard (no pin),
# two paired A/B rounds against CTL at all six loads (tools/ly_prtl_ab.sh, chk step first), then a decode 16 / 32
# kernel timeline of CAND (tools/ly_tl_run.py). Logs under .build/d16tl/ and .build/prtl/<TAG>{,_r2}/.
#   ssh okc-h200 'cd /root/okc && CAND=ly_d16un.py TAG=d16un nohup bash tools/ly_d16_ab.sh > .build/d16tl/d16un_run.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
CAND=${CAND:?CAND=file.py}; TAG=${TAG:?TAG=name}; CTL=${CTL:-ly_v1un.py}
COMMON=(--lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext
        --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json)
mkdir -p .build/d16tl
if [ -z "$NOGATE" ]; then
  echo "$(date -u +%T) gate start"
  python tools/bx_box_run.py muse_decode_layer "muse_decode_layer/cand/$CAND" --loads "decode 16,decode 32" --runs 20 \
    "${COMMON[@]}" --out ".build/d16tl/gate_$TAG.json" > ".build/d16tl/gate_$TAG.log" 2>&1
  echo "$(date -u +%T) gate rc=$?"
fi
CTL=$CTL CAND=$CAND PFX=$TAG STEPS="chk c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
CTL=$CTL CAND=$CAND PFX=${TAG}_r2 STEPS="c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
LY_TRACE_DIR=".build/d16tl/tl_$TAG" python tools/ly_tl_run.py muse_decode_layer "muse_decode_layer/cand/$CAND" \
  --loads "decode 16,decode 32" --runs 20 --kernels --pin-pdl8 1,8,16,32 --skip-pdl8-gate "${COMMON[@]}" \
  --out ".build/d16tl/tl_$TAG.json" > ".build/d16tl/tl_$TAG.log" 2>&1
echo "$(date -u +%T) all done"
