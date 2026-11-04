#!/bin/bash
# Decode 1 register slice on the box: reference check of the candidate at all six loads (chk), then the stamped
# decode 1 runs of the candidate's instrument and of the control's (tools/ly_box_pmt.py), every GPU step under
# gpu.lock. STEPS picks a subset.
#   ssh okc-h200 'cd /root/okc && CAND=ly_rsl.py PFX=rsl nohup tools/ly_rsl_box.sh > .build/rslc/rsl.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
D=.build/rslc/${PFX:-rsl}
mkdir -p "$D"
C=muse_decode_layer/cand
J=.build/judge/g1b4_pdl8d.f931abf568544018.json
SEED=106218864389449120
for step in ${STEPS:-chk pmt_w pmt_c}; do
  case $step in
    chk) python tools/bx_box_run.py muse_decode_layer "$C/${CAND:?}" --judge $J --router-seed $SEED --runs 5 \
           --loads "decode 1,decode 16,decode 32,verify 1,verify 16,verify 32" \
           --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
           --pin-pdl8 1,8,16,32 --skip-pdl8-gate --out "$D/chk.json" > "$D/chk.log" 2>&1 ;;
    pmt_w) python tools/ly_box_pmt.py "$C/${PMT_W:-ly_pmt_rsl.py}" --judge $J --router-seed $SEED --inputs 20 --raw 20 \
           --loads "decode 1" --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json \
           --out "$D/pmt_w.json" > "$D/pmt_w.log" 2>&1 ;;
    pmt_c) python tools/ly_box_pmt.py "$C/${PMT_C:-ly_pmt_v1un.py}" --judge $J --router-seed $SEED --inputs 20 --raw 20 \
           --loads "decode 1" --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json \
           --out "$D/pmt_c.json" > "$D/pmt_c.log" 2>&1 ;;
  esac
  echo "$(date -u +%T) $step rc=$?"
done
echo "$(date -u +%T) done"
