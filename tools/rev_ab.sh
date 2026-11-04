#!/bin/bash
# Revival lane on the box: for each "file.py:tag" in CANDS, a gate run at all six loads with the file's own prepare
# (in-prepare PDL8 / PDL8H guards active, reference check), then two paired A/B rounds against CTL through
# tools/ly_prtl_ab.sh. Every python step takes gpu.lock. Logs under .build/rev/ and .build/prtl/<tag>{,_r2}/.
#   ssh okc-h200 'cd /root/okc && CANDS="ly_rvo38.py:rvo38" nohup bash tools/rev_ab.sh > .build/rev/ab_run.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
CTL=${CTL:-ly_stk3.py}
mkdir -p .build/rev
for ct in ${CANDS:?CANDS=file.py:tag ...}; do
  CAND=${ct%%:*}; TAG=${ct##*:}
  echo "$(date -u +%T) $TAG gate start"
  python tools/bx_box_run.py muse_decode_layer "muse_decode_layer/cand/$CAND" \
    --loads "decode 1,decode 16,decode 32,verify 1,verify 16,verify 32" --runs 5 \
    --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
    --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
    --out ".build/rev/gate_$TAG.json" > ".build/rev/gate_$TAG.log" 2>&1
  rc=$?
  echo "$(date -u +%T) $TAG gate rc=$rc"
  grep -E "PDL8H_GUARD|PDL8_AB|Traceback|FAIL" ".build/rev/gate_$TAG.log" | head -20
  [ $rc -eq 0 ] || continue
  CTL=$CTL CAND=$CAND PFX=$TAG STEPS="chk c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
  CTL=$CTL CAND=$CAND PFX=${TAG}_r2 STEPS="c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
done
echo "$(date -u +%T) all done"
