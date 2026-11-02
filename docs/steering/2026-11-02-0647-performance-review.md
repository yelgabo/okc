# Architecture performance review, November 2 at 06:47 Vancouver time

Evidence snapshot `fcf5452`.

**The short serial pooler/router candidate passed correctness and was parked.** Private job `e4c8f00532b74d15` tested source `8c0bfba`, file hash `5d883c43`, against fresh G1b4 control `cc474883117947c0`, file hash `ba9e327c`. The serial candidate scored 2.093073× (95% CI 2.061677–2.124469) and the control scored 2.109201× (2.077563–2.140839). Each ran once on the same H200 node, passed 13 official tests, and produced 12 timing cells and profiles. The 56 startup replays in each job exercised the original startup path. [Terminal decision](https://github.com/yelgabo/okc/blob/fcf5452/docs/steering/2026-11-02-captain-serial-terminal-decision.md) · [Worker archive](https://github.com/yelgabo/okc/blob/fcf5452/docs/steering/2026-11-02-worker-short-serial-v2-terminal.md)

| Load | Tokens | Fresh control µs | Serial µs | Difference µs | Change |
|---|---:|---:|---:|---:|---:|
| d1 | 1 | 78.765 | 81.741 | +2.977 | +3.78% |
| d16 | 16 | 152.016 | 157.256 | +5.240 | +3.45% |
| d32 | 32 | 191.898 | 194.559 | +2.661 | +1.39% |
| v1 | 8 | 112.682 | 117.191 | +4.509 | +4.00% |
| v16 | 128 | 261.713 | 269.831 | +8.118 | +3.10% |
| v32 | 256 | 351.080 | 361.511 | +10.431 | +2.97% |

Four short-path signatures confirm that the one- and eight-token cells ran serially. All eight longer-cell kernel multisets matched. The table uses 46:16 sliding/full weights. The control-to-candidate latency factor across six loads was 0.969827×, below the 1.01 confirmation threshold, and every load exceeded the 1% regression limit. All 12 seeds differed across sessions. The longer cells also slowed despite using the same dispatch, so the cross-run timings cannot isolate the cost of serialization. G1b4 retains the public score of 2.126083×. [Comparison and selected paths](https://github.com/yelgabo/okc/blob/fcf5452/docs/steering/2026-11-02-captain-serial-terminal-decision.md)

**B1 is a separate 128/256-token buffer-lifetime study.** The plain-owner prototype stopped before candidate generation because synchronization alone could not guarantee the allocation would outlive a caller-owned graph. The CUDA graph user-object alternative has a conditional source-level ownership argument. H200 support, correctness, compression and timing are unmeasured. The last private check remains reserved. The B1 name refers to a buffer proposal and adds no staged grouped-MoE evidence to Family B. [Ownership decision](https://github.com/yelgabo/okc/blob/fcf5452/docs/steering/2026-11-02-captain-b1-graph-owner-discriminator.md) · [Independent lifetime review](https://github.com/yelgabo/okc/blob/fcf5452/docs/steering/2026-11-02-research-b1-user-object-proof.md)
