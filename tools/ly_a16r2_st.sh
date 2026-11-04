#!/bin/bash
# attention-d16 round 2 on the box: decode 16 / 32 attention stamps (tools/ly_box_a16.py, 20 inputs, cold L2) of each
# instrument file in ARMS, one at a time under gpu.lock. Logs: .build/a16/st_<arm>.log
#   ssh okc-h200 'cd /root/okc && ARMS="ly_stk3_st ly_aopf_st" nohup bash tools/ly_a16r2_st.sh > .build/a16/r2_st.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
COMMON=(--lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext
        --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json)
mkdir -p .build/a16
for a in ${ARMS:?ARMS="file_st ..."}; do
  echo "$(date -u +%T) stamps $a start"
  python tools/ly_box_a16.py "muse_decode_layer/cand/$a.py" "${COMMON[@]}" --inputs 20 --out ".build/a16/st_$a.json" \
    > ".build/a16/st_$a.log" 2>&1
  echo "$(date -u +%T) stamps $a rc=$?"
done
echo "$(date -u +%T) all done"
