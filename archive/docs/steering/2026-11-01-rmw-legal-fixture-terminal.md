# Corrected RMW pair: valid measurement, promotion closed

November 1 Pacific / November 2 UTC 2026. Evidence base f8a45afe886a17c880d44970c36346d6805e1daa; implementation/preregistration d5f29167b34f78db864d22550b23f4a0516f6a21; accepted-job archive cc80a2cf14403d4d00bac90ed8173a37b41c0c7f. Exact own PRIVATE **c5e4f89672f743bb**, replacement tranche **#3/4**.

## Decision

**Instrument PASS; frozen whole-six promotion gate FAIL. Close RMW-only promotion and do not spend #4.** The unchanged frozen reader exits 0, accepts complete evidence, and recomputes private_raw_gate=false. All three banks and all twelve bank/cross gains fall below both the registered 1% floor and twice duplicate-null. Bank993821 also fails the context regression gate. This is a valid subthreshold/mixed result, not another setup failure. Captain explicitly accepted this closure before publication. No remeasurement, selected-bank rescue, source tweak or public/scored action follows.

Official entry returned CURRENT repaired-common OFF. Its descriptive score is **2.112499365040427**, 95% CI **[2.0808118745648203, 2.144186855516033]**, score_us10707.837538008971. It is neither an isolated ON result nor a new promoted public candidate. The private raw cold-graph comparison supplies scoped OFF/ON attribution; the official score does not.

## Exact identity and collection

Accepted **2026-11-02 02:22:46.573432 UTC**, started **02:43:19.332235**, finished **02:48:30.159088**. Existing own wait11629 completed exit0; no new registration/restart or duplicate poll loop. Queue1232.758803s and worker310.826853s are separate from official setup178.253238s. The 295s guard applies to preparation, not queue or all official tests/benchmark/profile execution.

- Physical upload SHA256 **785a90ca08ddde0c3b194fc219a0645cb8cd14bac02620d8714f0b66163a0b66**, **924005bytes**.
- Frozen manifest SHA256 **c5f0e3975e30f4c012be8320615c7a19bbf9a7b7a63381906c10f3a3b3d4c574**,4821bytes.
- Frozen reader SHA256 **745d7bb508d170dbc6bcd4b86e0f8a0b39c9a9ba08885aec54a45a33e68f004d**,10920bytes.
- Device prefix **aea9735e7d91a7d51a7afe63c8cf0774b5bbd15de6ab416bbc5acbd2469ddbd2**; harness core **e5229e040d90ddb2f27ef1072f49d346aac87c6c123ef66f2ea4124e9f2f732f**.
- Common OFF base **4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280**; raw ON base **2884adb6d0d6947391c81cd67a0c727a04e7d41694d7c780cf8cee80dfa9dbcd**.
- Owning server content identity **28d360741906efb0ff5a90be98b92a053aff04480442d92992d47907ee557cab**, payload muse_decode_layer/28d360741906efb0ff5a90be98b92a053aff04480442d92992d47907ee557cab.py. Server and physical identities are distinct.

Raw JSON/log/txt are byte-preserved in muse_decode_layer/profiles/g-common-rmw-pair-v4.c5e4f89672f743bb.*. Python parses original JSON without losing 64-bit official seeds. The log attestation binds own ID, original output path, archived log hash and source. Reader/source/manifest stayed unchanged; derived calculations are separate from the frozen decision.

## Frozen matched result

Four independent mutable placements OFF0/OFF1/ON0/ON1, two code variants, original banks993811/993817/993821,36rows,24cold randomized rounds,256MiBflush, six loads1/8/16/32/128/256,46sliding:16full. Both arms retain common progress/cooperative/preexpert fixes. All four crosses are02/03/12/13. No historical607 or unrepaired control is run.

For each load C=(46*sliding+16*full)/62. Whole-six log gain is mean(log(C_OFF/C_ON)); positive means faster. Each bank/cross must exceed max(log1.01,2*duplicate log-null), and every load/context must regress by at most max(0.5us,1%). log1.01=0.009950330853; twice-null is0.007111095303/0.005663937815/0.007103379316 by bank. Thus the1%floor dominates for all three banks, but every observed gain also fails the null separation criterion.

| Bank | Log02 | Log03 | Log12 | Log13 | Duplicate log-null | Context regression | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 993811 | 0.000857999 | -0.000923355 | 0.002210247 | 0.000428893 | 0.003555548 | PASS | FAIL |
| 993817 | 0.002828144 | 0.001430787 | 0.002167329 | 0.000769972 | 0.002831969 | PASS | FAIL |
| 993821 | -0.001808081 | -0.001829079 | 0.001310492 | 0.001289494 | 0.003551690 | FAIL | FAIL |

The twelve whole-six factors span **-0.1827407% to +0.2832147%**. No bank/cross reaches1%. No confidence interval for these private median-derived gains is invented; the official score CI is a different statistic.

Short-only contributions below sum T1/T8 log gains with the same denominator6. They do not include the other four loads and are not whole-six promotion results.

| Bank | Short log02 /6 | Short log03 /6 | Short log12 /6 | Short log13 /6 |
| --- | ---: | ---: | ---: | ---: |
| 993811 | 0.000568697 | -0.000196012 | 0.001583975 | 0.000819265 |
| 993817 | 0.002324051 | 0.001491681 | 0.001868339 | 0.001035970 |
| 993821 | 0.000469814 | 0.000076208 | 0.001506296 | 0.001112690 |

Per-load range across all three banks/four crosses, reported as100*(C_OFF/C_ON-1):

| Tokens | Minimum speed gain | Maximum speed gain |
| --- | ---: | ---: |
| 1 | -0.2115% | 0.8596% |
| 8 | -0.1209% | 0.7211% |
| 16 | -0.6104% | 0.1664% |
| 32 | -0.4753% | 0.1447% |
| 128 | -0.1705% | 0.3583% |
| 256 | -0.3888% | 0.2371% |

All18 weighted load/bank rows, costs in us and cross gains in percent, retain all comparisons:

| Bank | T | OFF0 | OFF1 | ON0 | ON1 | Gains02 /03 /12 /13 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 993811 | 1 | 95.458 | 96.055 | 95.487 | 95.660 | -0.0303 / -0.2115 / 0.5946 / 0.4122 |
| 993811 | 8 | 122.349 | 122.332 | 121.896 | 122.234 | 0.3722 / 0.0942 / 0.3582 / 0.0802 |
| 993811 | 16 | 153.080 | 152.717 | 152.942 | 152.926 | 0.0898 / 0.1006 / -0.1475 / -0.1367 |
| 993811 | 32 | 207.157 | 207.514 | 207.214 | 208.146 | -0.0276 / -0.4753 / 0.1447 / -0.3038 |
| 993811 | 128 | 275.604 | 275.834 | 274.849 | 275.838 | 0.2747 / -0.0850 / 0.3583 / -0.0017 |
| 993811 | 256 | 376.537 | 377.230 | 377.150 | 376.444 | -0.1627 / 0.0245 / 0.0211 / 0.2087 |
| 993817 | 1 | 90.175 | 90.073 | 89.568 | 89.407 | 0.6782 / 0.8596 / 0.5641 / 0.7453 |
| 993817 | 8 | 133.303 | 133.090 | 132.349 | 133.251 | 0.7211 / 0.0391 / 0.5600 / -0.1209 |
| 993817 | 16 | 161.347 | 161.346 | 161.759 | 161.418 | -0.2549 / -0.0444 / -0.2556 / -0.0451 |
| 993817 | 32 | 190.682 | 190.794 | 190.580 | 190.871 | 0.0536 / -0.0990 / 0.1124 / -0.0403 |
| 993817 | 128 | 275.094 | 274.441 | 274.205 | 274.639 | 0.3243 / 0.1656 / 0.0862 / -0.0722 |
| 993817 | 256 | 358.116 | 358.318 | 357.471 | 358.326 | 0.1805 / -0.0585 / 0.2371 / -0.0020 |
| 993821 | 1 | 69.066 | 69.230 | 69.058 | 69.008 | 0.0120 / 0.0845 / 0.2496 / 0.3224 |
| 993821 | 8 | 86.542 | 86.875 | 86.309 | 86.575 | 0.2703 / -0.0387 / 0.6566 / 0.3464 |
| 993821 | 16 | 123.705 | 124.264 | 124.465 | 124.057 | -0.6104 / -0.2842 / -0.1613 / 0.1664 |
| 993821 | 32 | 162.519 | 163.429 | 163.279 | 163.216 | -0.4659 / -0.4272 / 0.0917 / 0.1306 |
| 993821 | 128 | 248.767 | 248.799 | 248.514 | 249.192 | 0.1022 / -0.1705 / 0.1151 / -0.1576 |
| 993821 | 256 | 341.949 | 342.726 | 343.284 | 342.839 | -0.3888 / -0.2595 / -0.1625 / -0.0330 |

The single failing context guard is **bank993821, T16, sliding, OFF0->ON0**:102.048002183us->103.199999779us, **+1.151997596us**, allowed1.020480022us, excess0.131517574us. This load's native treatment code is unchanged; the observed private-placement variation is not proof the short RMW causally slowed T16. It still fails the preregistered promotion rule and cannot be dropped after observing it. Even dropping this guard would not make any bank pass the gain/null gates.

## Instrument, source and native evidence

Official done/result.statusok, passed=true, return0, checks.ok=true, empty flags/rejections/error: **13tests/12benchmarks/12profiles PASS**. Frozen reader validates complete36row coverage,144initialmath +144fresh-final checks,8fresh replays per bank/placement,32empty checks including16occupied-to-empty checks,432actual native attention witnesses and actual expert routing. Startup prints56verified native replays. Distinct occupied IDs0..7 per token and copy_ restores retain captured tensor/storage identity and occupied->empty->occupied->empty coverage. The earlier v3 fixture violated the distinct-topk domain; success of this repaired fixture does not retroactively localize the old CUDA719.

All four placements own334995616mutable bytes each; logged free memory130912550912bytes. Source and CPU owner-negative checks were published before dispatch; GPU reader confirms common-control and cleanup_restored=true. Reference memo1008hits/144misses/peak4,1152total accesses. Actual resource vectors match across four placements and both bool entries. Cooperative admission is132CTAs on132SMs,1activeCTA/SM for every selected backend. Columns below come from the first common-resource tuple; both entry tuples are equal.

| Backend | Registers | Local bytes | Static shared B | Dynamic shared B | Block threads | Function max threads | Active CTA/SM | SMs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 145 | 0 | 0 | 214784 | 288 | 288 | 1 | 132 |
| tiny2 | 145 | 0 | 0 | 214784 | 288 | 288 | 1 | 132 |
| mid | 152 | 0 | 0 | 214784 | 288 | 288 | 1 | 132 |
| b4 | 96 | 8 | 0 | 231168 | 576 | 576 | 1 | 132 |
| main | 96 | 0 | 0 | 231168 | 576 | 576 | 1 | 132 |

Module flag vectors match registered policy: short relaxed final RMW only ON tiny/tiny2; common stage/progress fixes retained. MAIN/B4 select their expected compiled paths. These observations validate resources/admission/actual exercise; passing replays do not replace the retained ordering/progress proof. No native/production/timed math or memory-ordering edit was made for terminal evaluation.

## Preparation and official descriptive timings

| Host phase | Seconds |
| --- | ---: |
| normal_prepare | 106.454363564 |
| private | 0.692902553 |
| resources | 62.479975911 |
| empty | 0.026956612 |
| row | 8.596885325 |
| cleanup | 0.000124629 |
| Capsule total | 178.251211588 |
| Official setup total | 178.253237769 |

Capsule total178.251212s leaves116.748788s against295s. These host phase walls include work/waits in their existing phase boundaries; resource-phase62.479976s is not an isolated GPU kernel cost or proven pure compile time. Cleanup0.000124629s completed. Runtime fit is now observed for this exact run rather than inferred from c193 or failed v3.

Official adjusted medians below are repaired-common OFF only. Reconstructing the geometric mean of six weighted baseline/candidate ratios gives **2.112499365040427**, exactly the owning result. Private cold graph timings above have a different observation protocol and must not be substituted into official score_us.

| Load | Sliding us | Full us | Weighted candidate us | Weighted baseline us |
| --- | ---: | ---: | ---: | ---: |
| decode 1 | 77.202 | 84.198 | 79.008 | 175.007 |
| decode 16 | 133.527 | 214.589 | 154.446 | 305.743 |
| decode 32 | 152.522 | 307.092 | 192.411 | 353.717 |
| verify 1 | 117.311 | 121.933 | 118.503 | 270.196 |
| verify 16 | 243.715 | 332.470 | 266.619 | 566.376 |
| verify 32 | 313.435 | 478.179 | 355.950 | 810.243 |

## Handoff and stopping boundary

Fresh authenticated CLI snapshot **2026-11-02 02:53:24.842 UTC**: layer **7private/5scored remaining**,3checksused/0scoredused. Latest50jobs contain no live layer job; exact c5e4f89672f743bb is done/pass. Daily capacity is distinct from allocation: three tranche slots were used, and the conditional fourth heldout slot is **not eligible** after this gate failure. No new GPU action is authorized by remaining quota.

Worker owns archive/publication and then returns allocation to captain. Captain owns README/index and independent next-mechanism selection. No additional implementation is authorized before that selection. RMW source/pins and failed v3 artifacts remain available; no generic harness work or seed/mask/retry chase. The scoped closure covers this short final-RMW treatment in the current repaired-common layer on frozen matched banks/placements. A materially different mechanism with a new opportunity case would require a separate decision; source spelling or selected-bank rescue does not reopen it.

## Evidence

- [Frozen reader observation, all36rows and12bank/cross results](../../../docs/steering/evidence/2026-11-01-rmw-common-pair-v4/terminal-reader-observation.json)
- [Exact owning terminal receipt](../../../docs/steering/evidence/2026-11-01-rmw-common-pair-v4/own-terminal-receipt.json)
- [Terminal summary and timestamps](../../../docs/steering/evidence/2026-11-01-rmw-common-pair-v4/terminal-summary.json)
- [Derived weighted costs/regression/official reconstruction](../../../docs/steering/evidence/2026-11-01-rmw-common-pair-v4/terminal-derived.json)
- [All18weighted rows CSV](../../../docs/steering/evidence/2026-11-01-rmw-common-pair-v4/terminal-weighted-rows.csv)
- [Owning log attestation](../../../docs/steering/evidence/2026-11-01-rmw-common-pair-v4/terminal-log-attestation.json)
- [Raw profiles](../../../muse_decode_layer/profiles/g-common-rmw-pair-v4.c5e4f89672f743bb.json)
- [Preregistration, exact delta and CPU/source gates](2026-11-01-rmw-legal-fixture-registration.md)
- [Archive byte manifest](../../../docs/steering/evidence/2026-11-01-rmw-common-pair-v4/terminal-archive-manifest.json)

README is intentionally captain-owned; captain will index this report after the worker sends its pushed and remote-verified commit.
