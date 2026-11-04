#!/bin/bash
# Attention-v1 round 2 on the box: a gate run of CAND at verify 1 (the file's own in-prepare PDL8 guards, no pin), then
# tools/ly_prtl_ab.sh against CTL (default ly_stk1.py): the chk step (reference check at all six loads) and two paired
# rounds (c_a w_a c_b w_b c_c, 60 runs per arm, six loads). Logs under .build/av1r2/gate_<TAG>.log and
# .build/prtl/<TAG>{,_r2}/. Table: python3 tools/ly_prtl_table.py .build/prtl/<TAG> .build/prtl/<TAG>_r2
#   ssh okc-h200 'cd /root/okc && CAND=ly_aqw.py TAG=aqw setsid nohup tools/ly_av1_ab.sh > .build/av1r2/aqw_ab.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
CAND=${CAND:?CAND=file.py}; TAG=${TAG:?TAG=name}; CTL=${CTL:-ly_stk1.py}
mkdir -p .build/av1r2
echo "$(date -u +%T) gate start"
python tools/bx_box_run.py muse_decode_layer "muse_decode_layer/cand/$CAND" --loads "verify 1" --runs 20 \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --out ".build/av1r2/gate_$TAG.json" > ".build/av1r2/gate_$TAG.log" 2>&1
echo "$(date -u +%T) gate rc=$?"
CTL=$CTL CAND=$CAND PFX=$TAG STEPS="chk c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
CTL=$CTL CAND=$CAND PFX=${TAG}_r2 STEPS="c_a w_a c_b w_b c_c" tools/ly_prtl_ab.sh
echo "$(date -u +%T) all done"
