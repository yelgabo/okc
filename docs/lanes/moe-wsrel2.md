# Lane moe-wsrel2 checkpoint

Updated: 2026-11-04 15:45 PST (candidates built, box run launching)

## Goal

Single post-relay release at the ws_helper doneA credit (moe_helper0: TINY T=1, TINY2 T=8, MID T=16): drop the 32
per-lane pre-relay `__threadfence()` and have lane 0 fence after `__syncwarp()`, before its atomicAdds.
Acceptance: correct by the cited proof, no worse than ly_mpdl by more than 0.2 us at any sliding cell.

## Base

muse_decode_layer/cand/ly_mpdl.py sha256 a0ee638cf5342235a879054ac68da02e63d6f5e88f4834db6081e09937858a68

## Candidates (tools/dl_ly_wsrel1_patch.py; site matched once in MOE_CUDA_SRC, once in moe_helper0)

- muse_decode_layer/cand/ly_wsrel1.py (fence.acq_rel.gpu) sha256 323362a6afea3500f4c856e04361f7e5b8eb06cdbc68a65cdb88184bfd181ed5
- muse_decode_layer/cand/ly_wsrel1_sc.py (fence.sc.gpu) sha256 ef68943633ee037e16a4bab0bd4a971b3d7fce74782eec092ba937720e6d07a3
- muse_decode_layer/cand/ly_wsrel1_ab.py (A/B only; default arm = ly_mpdl helper, state wsrel_h ar1 / sc1)
  sha256 978bc1474cff880c0b91837522b5c5a7d20bdf5c3173cfe1f1ba49eb3f6b9d81

## Box

tools/ly_wsrel1_box.sh, PHASE="sass gate ab", outputs in /root/okc/.build/wsrel1/ (sass.log, chk_<cand>.log,
ab_r{1,2}.log). SASS: tools/ly_wsrel1_sass.py (nvcc device build of the helper source, TINY and MID define sets).

## Next action

Launch the box script, read sass.log, the gate logs and the A/B rounds.
