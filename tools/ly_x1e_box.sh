#!/bin/bash
# moe-t1 lane on the box: decode 1 MoE prologue stamps of CTL and CAND (ly_mtl3x instruments), the candidate's gate
# (its own prepare: PDL8_AB / PDL8H guards), then tools/ly_prtl_ab.sh: the chk step (reference check at six loads),
# and, only if every chk cell passes, two paired A/B rounds against CTL at all six loads. Every python step takes
# gpu.lock. Logs under .build/x1e/ and .build/prtl/<TAG>{,_r2}/.
#   ssh okc-h200 'cd /root/okc && CAND=ly_x1e.py TAG=x1e STC=ly_mtl3x_stk3.py STW=ly_mtl3x_x1e.py \
#     nohup bash tools/ly_x1e_box.sh > .build/x1e/x1e_run.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
CAND=${CAND:?CAND=file.py}; TAG=${TAG:?TAG=name}; CTL=${CTL:-ly_stk3.py}
COMMON=(--lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext
        --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json)
D=.build/x1e
mkdir -p $D
for f in $STC $STW; do
  python tools/ly_box_pmt.py "muse_decode_layer/cand/$f" "${COMMON[@]}" --loads "decode 1" --inputs 20 --raw 20 \
    --out "$D/st_${f%.py}.json" > "$D/st_${f%.py}.log" 2>&1
  echo "$(date -u +%T) stamps $f rc=$?"
done
[ -n "$STC$STW" ] && python tools/ly_mtl3x_summary.py $D/st_*.json > $D/st_summary_$TAG.txt 2>&1
if [ -z "$NOGATE" ]; then
  python tools/bx_box_run.py muse_decode_layer "muse_decode_layer/cand/$CAND" --loads "decode 1" --runs 20 \
    "${COMMON[@]}" --out "$D/gate_$TAG.json" > "$D/gate_$TAG.log" 2>&1
  echo "$(date -u +%T) gate rc=$?"
  grep -E "PDL8H_GUARD|PDL8_AB|Traceback|FAIL" "$D/gate_$TAG.log" | head -20
  ng=$(grep -E "^PDL8H_GUARD" "$D/gate_$TAG.log" | grep -c ": pass")
  if [ "$ng" != 5 ]; then echo "PDL8H guards passing: $ng of 5: no A/B"; exit 1; fi
fi
[ -n "$NOAB" ] && exit 0
CTL=$CTL CAND=$CAND PFX=$TAG STEPS="chk" tools/ly_prtl_ab.sh
npass=$(grep -c "check PASS" .build/prtl/$TAG/chk.log)
echo "$(date -u +%T) chk PASS cells: $npass"
grep -E "G_RELEASE_STARTUP_OK|CG_GATE" .build/prtl/$TAG/chk.log
if [ "$npass" != 12 ]; then echo "chk not 12/12: no A/B"; exit 1; fi
CTL=$CTL CAND=$CAND PFX=$TAG STEPS="c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
CTL=$CTL CAND=$CAND PFX=${TAG}_r2 STEPS="c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
python tools/ly_prtl_table.py .build/prtl/$TAG .build/prtl/${TAG}_r2 > .build/prtl/${TAG}_table_both_rounds.txt 2>&1
echo "$(date -u +%T) all done"
