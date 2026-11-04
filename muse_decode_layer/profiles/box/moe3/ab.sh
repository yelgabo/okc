#!/bin/bash
# moe lane round 3: gate (the file own in-prepare PDL8 guards, no pin), chk + two paired rounds against ly_stk1 at all
# six loads (tools/ly_prtl_ab.sh), then optional B4 / MID stamps of the candidate.
cd /root/okc && . tools/box_env.sh
CAND=${CAND:?}; TAG=${TAG:?}; CTL=${CTL:-ly_stk1.py}
C=(--judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext)
mkdir -p .build/moe3
echo "$(date -u +%T) gate start"
python tools/bx_box_run.py muse_decode_layer "muse_decode_layer/cand/$CAND" --loads "decode 16,decode 32" --runs 20 "${C[@]}" --out ".build/moe3/gate_$TAG.json" > ".build/moe3/gate_$TAG.log" 2>&1
echo "$(date -u +%T) gate rc=$?"
grep -E "PDL8H_GUARD|PDL8_AB|Traceback|Error" ".build/moe3/gate_$TAG.log" | head -20
CTL=$CTL CAND=$CAND PFX=m3_$TAG STEPS="chk c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
CTL=$CTL CAND=$CAND PFX=m3_${TAG}_r2 STEPS="c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
if [ -n "$TLB4" ]; then
  python tools/ly_box_moe_tlb4g.py "muse_decode_layer/cand/$TLB4" --loads "decode 32" --inputs 40 --raw 6 "${C[@]}" --out ".build/moe3/tlb4_$TAG.json" > ".build/moe3/tlb4_$TAG.log" 2>&1
  echo "$(date -u +%T) tlb4 rc=$?"
fi
if [ -n "$TL16" ]; then
  python tools/ly_box_moe_tl16g.py "muse_decode_layer/cand/$TL16" --loads "decode 16" --inputs 40 --raw 6 "${C[@]}" --out ".build/moe3/tl16_$TAG.json" > ".build/moe3/tl16_$TAG.log" 2>&1
  echo "$(date -u +%T) tl16 rc=$?"
fi
echo "$(date -u +%T) all done"
