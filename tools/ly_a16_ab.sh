#!/bin/bash
# attention-d16 lane on the box: gate run of CAND at decode 16 / 32 with the file's own prepare (PDL8 / PDL8H guards
# unpinned), two paired A/B rounds against CTL (default ly_stk2.py) at all six loads through tools/ly_prtl_ab.sh (chk
# step first), then (STAMP=file.py) the decode 16 / 32 attention stamps of the candidate's instrument build.
# Every python step takes gpu.lock. Logs under .build/a16/ and .build/prtl/<TAG>{,_r2}/.
#   ssh okc-h200 'cd /root/okc && CAND=ly_a16kv.py TAG=a16kv STAMP=ly_a16kv_st.py nohup bash tools/ly_a16_ab.sh > .build/a16/a16kv_run.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
CAND=${CAND:?CAND=file.py}; TAG=${TAG:?TAG=name}; CTL=${CTL:-ly_stk2.py}
COMMON=(--lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext
        --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json)
mkdir -p .build/a16
if [ -z "$NOGATE" ]; then
  echo "$(date -u +%T) gate start"
  python tools/bx_box_run.py muse_decode_layer "muse_decode_layer/cand/$CAND" --loads "decode 16,decode 32" --runs 20 \
    "${COMMON[@]}" --out ".build/a16/gate_$TAG.json" > ".build/a16/gate_$TAG.log" 2>&1
  echo "$(date -u +%T) gate rc=$?"
  grep -E "PDL8H_GUARD|PDL8_AB|Traceback|FAIL" ".build/a16/gate_$TAG.log" | head -20
fi
if [ -n "$STAMP" ]; then
  python tools/ly_box_a16.py "muse_decode_layer/cand/$STAMP" "${COMMON[@]}" --inputs 20 --out ".build/a16/st_$TAG.json" \
    > ".build/a16/st_$TAG.log" 2>&1
  echo "$(date -u +%T) stamps rc=$?"
fi
CTL=$CTL CAND=$CAND PFX=$TAG STEPS="chk c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
CTL=$CTL CAND=$CAND PFX=${TAG}_r2 STEPS="c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
echo "$(date -u +%T) all done"
