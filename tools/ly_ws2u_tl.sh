#!/bin/bash
# MID timeline (decode 16, 40 inputs per cell) of the control and the ly_ws2u variants, one process each, every one
# taking gpu.lock in-process (tools/ly_box_moe_tl16.py --lock). Files: ly_tl16.py (tools/ly_moe_tl16.py) and
# ly_ws2u{a,b}_tl.py (tools/dl_ly_ws2u_patch.py --tl).
#   ssh okc-h200 'cd /root/okc && nohup tools/ly_ws2u_tl.sh > .build/ly2u/tl.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
mkdir -p .build/ly2u
for f in ${FILES:-ly_tl16 ly_ws2ua_tl ly_ws2ub_tl}; do
  python tools/ly_box_moe_tl16.py "muse_decode_layer/cand/$f.py" \
    --judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --loads "decode 16" \
    --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
    --out ".build/ly2u/tl_$f.json" > ".build/ly2u/tl_$f.log" 2>&1
  echo "$(date -u +%T) $f rc=$?"
done
