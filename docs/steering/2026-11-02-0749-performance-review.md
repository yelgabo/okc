# Architecture review, November 2 at 07:49 Vancouver time

Snapshot `218a048`.

The graph-owned PR buffer candidate passed host syntax and CPU checks at worker source `178c772` (881,883 bytes, SHA256 `4c6d3f13e0b9c72c4d4ab67f440a35f6ae39a954c818f78d35f3ff3d9090423b`). The earlier cleanup protocol stopped at a source-level capture conflict. Its replacement waits for the buffer owner’s completion events and restores the helper’s prior capture mode even on errors. [Cleanup finding](https://github.com/yelgabo/okc/blob/218a048/docs/steering/2026-11-02-captain-pr-cleanup-capture.md) · [Host qualification and allocation](https://github.com/yelgabo/okc/blob/218a048/docs/steering/2026-11-02-captain-pr-host-gate.md)

One private layer check is conditionally allocated. The worker must first add a preparation receipt that identifies actual owner-tagged PR plans at 128 and 256 tokens. At this snapshot there is no dispatch ID, H200 correctness result or timing for the candidate. A run without the intended plans would establish no treatment. The fresh G1b4 control is `cc474883117947c0`. Any future comparison against that run would be cross-run. [Host qualification and allocation](https://github.com/yelgabo/okc/blob/218a048/docs/steering/2026-11-02-captain-pr-host-gate.md)

This buffer proposal is separate layer context. The A-versus-B decision still rests on the existing MoE comparisons. Public G1b4 retains 2.126083×.
