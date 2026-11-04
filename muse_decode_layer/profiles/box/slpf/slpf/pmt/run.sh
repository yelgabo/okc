cd /root/okc && . tools/box_env.sh
for f in ly_pmt_v1un ly_pmt_slpf ly_pmt_slpfb; do
  python tools/ly_box_pmt.py muse_decode_layer/cand/$f.py --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
    --router-seed 106218864389449120 --inputs 20 --lock /root/okc/.build/box/gpu.lock \
    --ls-cache /root/okc/.build/box/ls_cache.json --out .build/prtl/slpf/pmt/$f.json --raw 20 > .build/prtl/slpf/pmt/$f.log 2>&1
  echo "$(date -u +%T) $f rc=$?"
done
