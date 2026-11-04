#!/bin/bash
# Attention verify 1 lane on the box: the stamped candidate's attention timeline (tools/ly_box_atst.py, sliding),
# then tools/ly_v1_ab.sh (gate, chk + two paired A/B rounds against CTL, kernel timeline).
#   ssh okc-h200 'cd /root/okc && CAND=ly_apw.py STAMP=ly_atst_apw.py TAG=apw setsid nohup tools/ly_atw_run.sh > .build/atst/apw_run.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
CAND=${CAND:?CAND=file.py}; TAG=${TAG:?TAG=name}; CTL=${CTL:-ly_v1un.py}
mkdir -p .build/atst
if [ -n "$STAMP" ]; then
  python tools/ly_box_atst.py "muse_decode_layer/cand/$STAMP" --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
    --router-seed 106218864389449120 --inputs 20 --raw 2 --layers "${LAYERS:-sliding}" \
    --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
    --out ".build/atst/atst_$TAG.json" > ".build/atst/atst_$TAG.log" 2>&1
  echo "$(date -u +%T) atst_$TAG rc=$?"
fi
[ -n "$STAMPONLY" ] && exit 0
CAND=$CAND CTL=$CTL TAG=$TAG tools/ly_v1_ab.sh
