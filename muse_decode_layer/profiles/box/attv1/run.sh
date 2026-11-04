cd /root/okc && . tools/box_env.sh
for i in 2 3 4 5; do
  python tools/att_v1_cmp.py muse_decode_layer/cand/ly_micro2.py .build/repeats/attention_base.py --runs 60 --seed $((880001 + 100000 * i)) --lock /root/okc/.build/box/gpu.lock --ext-cache /root/.cache/bx_ext --out .build/box/attv1/p$i.json > .build/box/attv1/p$i.log 2>&1
done
python tools/bx_box_run.py muse_decode_layer muse_decode_layer/cand/ly_micro2.py --judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --runs 60 --loads "verify 1" --lock /root/okc/.build/box/gpu.lock --ls-cache /root/.cache/../okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext --pin-pdl8 1,8,16,32 --skip-pdl8-gate --kernels --out .build/box/attv1/layk.json > .build/box/attv1/layk.log 2>&1
echo ALLDONE >> .build/box/attv1/p5.log
