# Published attention file on the official judge

All values are adjusted per-cell judge medians in µs. Each column uses an independent H200 run with its own inputs and SGLang baseline. The private study used only the external file as an off-board comparator.

| Load | Context | Owned public 8914751e3b9f468b | Owned control f74cbe27819147c1 | Published file cd6805e7ebce45ee |
|---|---|---:|---:|---:|
| decode 1 | sliding | 5.183 | 5.168 | 4.673 |
| decode 1 | full | 12.390 | 12.654 | 10.794 |
| decode 16 | sliding | 7.833 | 8.196 | 8.014 |
| decode 16 | full | 85.046 | 84.413 | 86.465 |
| decode 32 | sliding | 10.092 | 10.292 | 10.247 |
| decode 32 | full | 162.552 | 162.156 | 166.089 |
| verify 1 | sliding | 6.770 | 7.115 | 6.356 |
| verify 1 | full | 14.819 | 14.786 | 14.232 |
| verify 16 | sliding | 11.454 | 11.617 | 11.105 |
| verify 16 | full | 101.555 | 101.663 | 100.722 |
| verify 32 | sliding | 15.603 | 15.299 | 14.634 |
| verify 32 | full | 183.564 | 183.016 | 184.146 |

Source SHA256 394f2517ca8eaf69b7d422832edfd59451a9f73103981d13b1059070abb17b4b matched the public artifact. The official result passed 16 tests at 2.639606424× (95% CI 2.600012327–2.679200520). The published study reports 2.6537× and assigns the same candidate/baseline ratio to every tested case, with baseline times increasing by 15 µs per case index. Its JSON cannot show which shape improved. [Published source and evaluation](https://github.com/rohanarun/Auto-research-agent/tree/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks/data/candidates/gemini-3.8-flash-opencode/kernel-muse_attention_decode).

The decode-1 46/16 mix in isolation accounts for a 1.020034× six-load counterfactual against the owned public receipt. Taking the lower observed median in each of the 12 cells gives an optimistic 1.033593× factor. These are cross-run sensitivities. An accepted hybrid or a causal speedup would require a matched experiment and independently owned code. The external point estimate still needs 2.64% for the rank-50 score in this snapshot.
