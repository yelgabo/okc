#!/bin/bash
# bf16 lane, qkv_gate at T = 16 / 32 on the box: numerics of CAND against the wide build, per-CTA qkv stamps of the
# control (ly_qtl_stk2.py) and of CAND's instrumented twin (QTL_CAND, narrow) at decode 16 / 32, then CAND's gate run,
# chk and two paired A/B rounds against ly_stk2 (tools/ly_d16_ab.sh). Every GPU process takes gpu.lock itself.
#   ssh okc-h200 'cd /root/okc && CAND=ly_q32n.py QTL_CAND=ly_qtl_q32n.py TAG=q32n nohup bash tools/ly_q32_run.sh > .build/q32/q32n_run.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
CAND=${CAND:?}; TAG=${TAG:?}
D=.build/q32; mkdir -p $D
L=(--lock /root/okc/.build/box/gpu.lock)
Q=(--judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --inputs 10
   --ls-cache /root/okc/.build/box/ls_cache.json "${L[@]}")
C=muse_decode_layer/cand
if [ -z "$NONUM" ]; then
  python tools/ly_q32_numerics.py $C/$CAND "${L[@]}" $NUMARGS > $D/num_$TAG.log 2>&1; echo "$(date -u +%T) num rc=$?"
fi
if [ -z "$NOSTAMP" ]; then
  for t in 16 32; do
    [ -n "$NOBASE" ] || { python tools/ly_box_qtl.py $C/ly_qtl_stk2.py "${Q[@]}" --loads "decode $t" --tokr $t \
      --out $D/st_base_$t.json > $D/st_base_$t.txt 2>&1; echo "$(date -u +%T) st base $t rc=$?"; }
    [ -z "$QTL_CAND" ] || { python tools/ly_box_qtl.py $C/$QTL_CAND "${Q[@]}" --loads "decode $t" --tokr $t --narrow \
      --out $D/st_${TAG}_$t.json > $D/st_${TAG}_$t.txt 2>&1; echo "$(date -u +%T) st $TAG $t rc=$?"; }
  done
fi
[ -n "$NOAB" ] || CTL=ly_stk2.py CAND=$CAND TAG=$TAG bash tools/ly_d16_ab.sh
echo "$(date -u +%T) q32 run done"
