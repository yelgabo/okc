#!/bin/bash
# verify 1 finish spread: v1st stamps of the env-controlled instrument ly_v1fsst.py at several finish reservations
# (V1FS_SMEM bytes; 0 keeps Triton's 4,096, the base's launch). Logs under .build/v1fs/.
#   ssh okc-h200 'cd /root/okc && SIZES="0 118784 98304" nohup tools/ly_v1fs_sweep.sh > .build/v1fs/sweep.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
mkdir -p .build/v1fs
INST=${INST:-ly_v1fsst.py}
for S in ${SIZES:?SIZES="0 118784"}; do
  echo "$(date -u +%T) smem $S start"
  V1FS_SMEM=$S python tools/ly_box_v1st.py "muse_decode_layer/cand/$INST" --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
    --router-seed 106218864389449120 --inputs 20 --raw 20 --lock /root/okc/.build/box/gpu.lock \
    --ls-cache /root/okc/.build/box/ls_cache.json --out ".build/v1fs/st_${S}${SFX}.json" > ".build/v1fs/st_${S}${SFX}.log" 2>&1
  echo "$(date -u +%T) smem $S rc=$?"
done
