# Transparent production hybrid fits; eligibility remains blocked

2026-11-01. Continuation from root `65f8810`, with source factories reviewed
against subsequent tracked root tools. This is a **source-size and source
equivalence milestone**, not GPU/public readiness or a performance result.
No GPU execution, official call or public submission was made by this worker.

The prior representation's NOFIT result remains valid historically. Its
single verify32 file was 1,072,104 bytes and all-four file 1,125,548 bytes.
The newly tested representation removes nonessential comments, preserving
readable code, source-generation anchors and author/license/NOTICE blocks.
It fits without constexpr pruning, docstring removal or blank-line removal.
Only masks 4 and 15 were built; the sixteen-mask matrix was not rerun.

## Frozen files and exact budget

Under `.build/today-proj/prod-size/frozen/`:

| File | Bytes | SHA256 |
|---|---:|---|
| `hybrid-mask4.py` | 907642 | `3b205d408edd8a238dec686fac306b5a28ac9136b42512939ae80db7c79611ec` |
| `hybrid-mask15.py` | 951539 | `d3f2b31ea0991e8b7e14709913e8a55d1304fe20fc47fd369e8a5f1c0334d689` |
| `G-comment.py` | 620061 | `0fadbabece84c447b5c197ee23372c19db8544055494d3dc3f0eb430d059cce2` |
| `owned-4-comment.py` | 282153 | `7612276301d5da296d35995f852b51328f6f33d20c0fd7b9c17bec5fbfe8656c` |
| `owned-15-comment.py` | 326021 | `c1c6065a095efb407a15c8a1fe8f6936fc3078750b1ca7adb27447846342205e` |

Mask 4 selects verify32/sliding only; mask 15 selects all four verify16/32
sliding/full cells. The generator reuses the existing fixed dispatcher and
eligibility/fallback body. Mask zero remains the existing generator's exact
original `607eb52bdaf24160` source; no default or production kernel file was
modified. Existing private comparison packages are unchanged.

| Component | Previous single32 | Packed single32 | Previous all4 | Packed all4 |
|---|---:|---:|---:|---:|
| Complete factored G | 720525 | 620061 | 720525 | 620061 |
| Owned selected backend | 345746 | 282153 | 399155 | 326021 |
| Readable embedding/dispatcher overhead | 5833 | 5428 | 5868 | 5457 |
| **Total** | **1072104** | **907642** | **1125548** | **951539** |

The strict cap is below 1,000,000 bytes: single32 has 92,357 bytes spare,
all4 has 48,460. Those are exact serialized UTF8 sizes, including code,
notices, source literals and dispatcher. No encoded/compressed source,
embedded cubin, opaque reconstruction or runtime source-file I/O is used.
Existing explicit named raw C++ fragment factoring is retained.

## Specific byte plan

The initial bounded constexpr-body inventory found 23,756 bytes in owned
ONE and 11,128 in owned MoE: only 34,884 raw body bytes before annotations.
Even that optimistic saving would leave single32 at least 1,037,220 bytes.
That plan failed the required gate and was **not implemented**. It is not
a universal lower bound on every possible source representation.

The accepted plan strips comments from complete native literals **before**
applying the existing G factor. It never lexes an arbitrary shared fragment
that could begin inside a comment/string. Then it removes nonprotected Python
COMMENT tokens. Python code/AST, docstrings, blank lines and code indentation
are retained. Native strings/chars/raw strings and every existing newline
are retained; replaced comments leave separating whitespace. This avoids
joining tokens such as `a/**/b`. Multiline preprocessor comments are retained.
Splice-created comment delimiters are rejected rather than approximated.

Exact pre-factor G removal is 125,970 bytes: 117,606 native comment bytes
and 8,364 Python-comment bytes. After refactoring, the **net** G saving is
100,464 bytes; common comments already shared by the original factor cannot
be counted twice. Largest raw native contributions are ATT7 48,336 bytes,
MoE main 28,015, helper0 24,663 and SK2 5,831. Full named per-literal counts
are in `source-manifest.json`.

Owned32 removes 62,535 native-comment bytes and 723 Python-comment bytes;
335 further bytes come from the existing readable raw-literal renderer,
giving 63,593 total. Owned-all4 removes 71,988 native and 811 Python-comment
bytes plus the same 335 representation bytes, totaling 73,134. Embedding
overhead changes by 405/411 bytes respectively. Code values remain readable
multiline literals; this is not code minification or a codec.

The G inventory retains 16,312 protected Python-comment bytes and 14,062
docstring bytes, plus all its blank lines. The owned inventories likewise
retain docstrings/blank lines; no speculative savings from them are needed.
Smaller unchanged native declarations and generated inline fragments receive
no assumed byte saving. Final complete-file measurements settle the fit.

## Notices, source assembly and corrected provisional failures

The guard preserves entire adjacent comment groups containing legal/NOTICE
markers, URLs, source/author credits and the named contributor terms required
by root (including bet4_claude, alygg, vjeux, keell0renz, matvey, Nathen,
chen, Zhimao, SGLang and Marlin). The full original license/NOTICE trailer is
retained. Marker comments and comments intersecting runtime text anchors are
also preserved. Non-native AST string constants are inspected so local named
anchors are not missed.

Two **invalid preliminary representations** were caught before any GPU use:
one removed `//@MOE1_DISPATCH@`, suppressing owned MAIN51 dispatch generation;
another missed GLE's named anchor containing `// every thread`, causing the
actual `_gle_source` assertion to fail. Earlier 947129/951226/951242-byte
files and their pins are superseded and must not be used. Native token proof
and device-only SASS alone would not have caught these host builders.

MoE review `d5acbba29127c87f51c627866754c899741e2d81` executes real module
imports and native source factories with explicit SGLang/Triton/capability/
load_inline CPU stubs. It captures 13 G build requests: ATT7, BFP, graft glue,
b2, sliding cluster, DE, router, GLE and all five expert build keys through
the actual M1/8/16/32/128/256 state policy. It also captures four owned32 and
five owned-all4 builders: MoE, ONE, route/boundary, Lt and WD where selected.
All CPP/CUDA noncomment streams, flags, exports and registrations match.
Only explicitly whitelisted MD5-derived DE/GLE identifier suffixes differ.
Owned `_moe1_source` emits exactly one case 10 selecting MAIN51.
All 204 G Python definitions/imports and 56/58 owned definitions/imports
match the originals. The full factory ledger is retained with the evidence.

Six owner fixtures pass, including independent Zig/Clang preprocessed-token
comparisons for strings, raw strings, escaped quotes, character literals,
token adjacency/stringification, preprocessor lines and `__LINE__`; phase-2
unsupported-form rejection; legal/credit preservation; Python AST/docstrings;
Unicode offsets; and executed named-anchor/dispatch-marker expansion. The
separate auditor's actual-corpus lexer/notice review is `18feac4`; its scope
and later source links remain separate from a runtime eligibility decision.

## Bounded local compiler proof

CUDA12.8 NVRTC/PTXAS sm90a checks compare original/comment device code only.
They do not link Torch host extensions or run CUDA kernels. Existing standard
header, infinity, host-boundary and declaration-only adapters are disclosed
in every compiler manifest. Ordinary O3 and actual selected defines are used.

G: five actual build keys, each with both bool instantiations:

| Build | Instructions per entry | Registers | Barriers | Stack / spill stores / spill loads B |
|---|---:|---:|---:|---|
| Main | 24112 | 96 | 16 | 8 / 4 / 32 |
| B4 | 26984 | 96 | 16 | 16 / 120 / 216 |
| Tiny | 5272 | 147 | 3 | 0 / 0 / 0 |
| Tiny2 | 5328 | 147 | 3 | 0 / 0 / 0 |
| Helper/mid | 5304 | 155 | 3 | 0 / 0 / 0 |

All selected instructions/operands and **all emitted out-of-line text
functions** match after self-symbol/branch-target normalization. Actual
resource lines and PTX launch bounds also match. Existing main/B4 spills
are retained; no new spill is introduced. Exact full CUDA source byte hashes
and effective defines match the real captured factory requests, linking the
compiled cases to these final artifacts rather than an obsolete draft.

Owned review `21ef9f3` and final marker/device-byte link `6366de8` compare
L8/mode3 (2328 instructions, 114 registers), L8/mode51 (6344, 80 registers)
and MoE MAIN51 (17528, 96 registers). All have 16 barriers and zero stack/
spill bytes; selected normalized SASS and resources match. ONE uses actual
`-DONE_TRIG=1`. Restoring the host marker did not change these compiled
device input bytes. This covers named entries, **not every G or owned kernel**.
Other native families have source-factory/token equivalence, not a newly
claimed exhaustive compile or H200 execution gate.

## Release blocker and next action

**These exact files are not GPU/public-ready.** The separate owned progress
audit `cc067bc`/`6366de8` identifies an inherited ordinary/PDL-only MAIN51
software-grid join without a documented whole-grid progress guarantee and
timeout fallthrough. Comment packing preserves that code; it does not repair
it. Root has assigned a separate cooperative/PDL-off/fail-closed source epoch.
The four-cell qualification is held for that repair. Do not transfer the
source-size result to performance eligibility.

Next: compose and independently review that progress repair, regenerate exact
sizes/pins and source-factory/compile proofs for the new epoch, then prepare
only a single-arm qualifier if its helpers fit. Actual host linking, resource
admission, complete G plus owned preparation below 300 seconds (295-second
guard), fresh inputs/caches/routing, official correctness and full-six-load
performance are still open. No private comparator result or conditional
two-load score prediction establishes a winning production hybrid.

Stop if a required runtime source anchor cannot be preserved within the cap,
factory/notice equivalence fails, the progress contract remains unsupported,
or numerical/prepare/resource gates fail. No precision, fence, launch-policy
or tuner change was made to achieve these byte counts. No performance gain
or new public candidate is recommended by this packaging checkpoint.

Reproduce from the projection checkout:

```text
python tools/prod_size_comments.py ../okc .build/today-proj/prod-size/comment-budget.json --emit .build/today-proj/prod-size/frozen
python tools/prod_size_test_comments.py -v
```

The manifest pins all eight emitted source files and six unchanged shared
generator inputs. Compiler manifests, logs, normalized disassembly evidence,
actual builder links and named byte counts are under
`docs/steering/evidence/2026-11-01-production-comment-fit/`. Compressed
disassembly is evidence only and is never included or read by the payload.
