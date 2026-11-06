"""Build muse_decode_layer/final/decode_layer_fixed.py from the judged final file.

The judged file stays byte-identical. Each edit must match its anchor the stated number of
times or the script stops; docs/decode-layer-final-review.md says what each edit fixes.

    python3 tools/dl_final_fix_patch.py [--check]
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/final/decode_layer.py"
DST = ROOT / "muse_decode_layer/final/decode_layer_fixed.py"
JUDGED_SHA256 = "1880297e7eaf4c7bb68711d0cca3cf38ca29ad08bdf05538577b8315e5b965d2"
SIZE_LIMIT = 1_000_000


def normalize_whitespace(text):
    out = []
    for line in text.split("\n"):
        stripped = line.rstrip(" \t")
        # A blank line right after a backslash continuation ends the macro; keep it, emptied.
        if line and not stripped and not (out and out[-1].endswith("\\")):
            continue
        out.append(stripped)
    return "\n".join(out)


def string_constants(text):
    return [n.value for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


# Constants that are only ever replacement text, never searched for, so a changed count is harmless.
REPLACEMENT_ONLY = {
    # _gmb_trig inserts it into glue edit strings; it newly matches ATT7's pdl_launch_dependents() body.
    '{\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n',
}


def check_anchors_survive(before, after):
    """Every string constant must occur in every large string constant as often after the
    whitespace pass as before, so no source patch anchor gains or loses a match."""
    a, b = string_constants(before), string_constants(after)
    if len(a) != len(b):
        raise SystemExit(f"constant count changed: {len(a)} -> {len(b)}")
    hay = [(x, y) for x, y in zip(a, b) if len(x) >= 200]
    needles = {(x, y) for x, y in zip(a, b)
               if len(x) >= 8 and (" " in x or "\n" in x) and x not in REPLACEMENT_ONLY}
    bad = []
    for x, y in needles:
        for hx, hy in hay:
            if hx.count(x) != hy.count(y):
                bad.append((x[:60], hx.count(x), hy.count(y)))
                break
    if bad:
        raise SystemExit("anchor counts changed by the whitespace pass:\n" + "\n".join(map(repr, bad[:20])))
    return len(needles), len(hay)


FRESH_HELPERS = '''FRESH_ROUNDS = 32


def _fresh_rounds(arms, data, gen, rounds):
    dev = data["stream"].device
    z = lambda: {"bits": torch.zeros((), dtype=torch.int64, device=dev), "ints": torch.zeros((), dtype=torch.int64, device=dev),
                 "rel": torch.zeros((), device=dev)}
    acc = {"off": z(), "on": z(), "cross": z()}
    s_buf, x_buf = data["stream"], data["attn_input"]
    for r in range(rounds):
        data["local_start"].fill_(32 * (r % 8))
        inputs = []
        for _ in range(2):
            s = torch.randn(s_buf.shape, generator=gen, device=dev, dtype=s_buf.dtype)
            inputs.append((s, (s.float() * torch.rsqrt(s.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(x_buf.dtype)))
        settled = {}
        for arm, (g, o) in arms.items():
            for i, (s, x) in enumerate(inputs):
                outs = []
                for _ in range(2):
                    s_buf.copy_(s)
                    x_buf.copy_(x)
                    g.replay()
                    outs.append([t.clone() for t in o])
                _mpdl_cmp(outs[0], outs[1], acc[arm])
                settled[arm, i] = outs[1]
        for i in range(2):
            _mpdl_cmp(settled["on", i], settled["off", i], acc["cross"])
        _mpdl_wait(f"fresh-input round {r}")
    return {k: {n: (float(v) if n == "rel" else int(v)) for n, v in a.items()} for k, a in acc.items()}


def _fresh_guard(st, T, layer, g_off, o_off, g_on, o_on, data, gen, cross, tag="PDL8"):
    # Each checked replay follows one on different inputs (the arms share module-level scratch), so a
    # read that ran ahead of its producer sees stale values and differs from the next replay on the same inputs.
    a = _fresh_rounds({"off": (g_off, o_off), "on": (g_on, o_on)}, data, gen, FRESH_ROUNDS)
    base = a["off"]
    limit = PDL8H_IDS_MULT * base["ints"] + PDL8H_IDS_SLACK
    bound = max(PDL8H_BOUND, base["rel"])
    ok = all(c["ints"] <= limit and c["rel"] <= bound for c in [a["on"]] + ([a["cross"]] if cross else []))
    print(f"{tag} FRESH T={T} {layer}: {FRESH_ROUNDS} rounds x 2 inputs; off self ids {base['ints']} rel {base['rel']:.2e}; "
          f"on self ids {a['on']['ints']} rel {a['on']['rel']:.2e}; on vs off ids {a['cross']['ints']} "
          f"rel {a['cross']['rel']:.2e}{'' if cross else ' (logged only)'} -> {'pass' if ok else 'FAIL'}", flush=True)
    return ok


def _mpdl_layer(st, T, layer, gen, dev):
'''

EDITS = [
    ("header", "# ly_bfq: ly_stk3 with qkv at T = 128",
     "# decode_layer_fixed: eat_b4 with the prepare-time guard and fallback fixes of tools/dl_final_fix_patch.py\n"
     "# (docs/decode-layer-final-review.md); CUDA kernels unchanged apart from whitespace. Not judged.\n"
     "# eat_b4: ly_stk7c + EAT best_4: glue-boundary cluster reductions by st.async into mbarriers, and the\n"
     "# boundary's PDL trigger at kernel entry (GMB_TRIG).\n"
     "# The harness/*.py scripts named below are not in this repository; tools/dl_pdl8e_cgemv.py is on branch barriers.\n"
     "# ly_bfq: ly_stk3 with qkv at T = 128", 1),
    ("stack_df1 header",
     "harness/make_dl_df.py (moe-small E173).\n",
     "harness/make_dl_df.py (moe-small E173).\n"
     "#   Later ruled out (docs/ruled-out.md, OKC_DF): the MID build compiles moe_helper0 without it.\n", 1),
    ("budget and patch helpers", "\nPDL_MAX_T = 128\n",
     "\nPDL_MAX_T = 128\n"
     "PREP_BUDGET_S = 300.0\n"
     "# Estimates, not measurements: room for one optional stage's graphs and replays; BFQ's also covers its compile.\n"
     "PREP_STAGE_MIN_S = 30.0\n"
     "PREP_BFQ_MIN_S = 90.0\n"
     "_PREP_T0 = None\n\n\n"
     "def _prep_left():\n"
     "    import time\n"
     "    return PREP_BUDGET_S if _PREP_T0 is None else PREP_BUDGET_S - (time.monotonic() - _PREP_T0)\n\n\n"
     "def _patch(s, old, new, n=1):\n"
     "    if s.count(old) != n:\n"
     "        raise RuntimeError(f\"source anchor {old[:60]!r}: {s.count(old)} found, {n} expected\")\n"
     "    return s.replace(old, new)\n", 1),
    ("workspace capture check",
     '        assert not torch.cuda.is_current_stream_capturing(), "attention rows must be warmed before capture"\n',
     '        if torch.cuda.is_current_stream_capturing():\n'
     '            raise RuntimeError("attention rows must be warmed before capture")\n', 1),
    ("attention self-test fallback",
     '    if st.get("att_ok"):\n        _attn_selftest(st)\n    _check_guards()\n',
     '    if st.get("att_ok"):\n'
     '        try:\n'
     '            _attn_selftest(st)\n'
     '        except RuntimeError as ex:\n'
     '            print("attention self-test failed, FA3 path:", repr(ex)[:200], flush=True)\n'
     '            st["att_ok"] = False\n'
     '            torch.cuda.synchronize()\n'
     '    _check_guards()\n', 1),
    ("native build fallback",
     "            except RuntimeError as exc:\n                st['_fork_build_errors'][m] = str(exc)[-300:]\n",
     "            except Exception as exc:\n                st['_fork_build_errors'][m] = str(exc)[-300:]\n", 1),
    ("pdl8 budget",
     "            continue\n        try:\n            data = _pdl8_data(B, 1, layer, gen, dev)\n",
     "            continue\n"
     "        if _prep_left() < PREP_STAGE_MIN_S:\n"
     "            st[\"pdl8_on\"][T] = False\n"
     "            print(f\"PDL8 T={T} off: prepare budget, {_prep_left():.0f} s left\", flush=True)\n"
     "            continue\n"
     "        try:\n            data = _pdl8_data(B, 1, layer, gen, dev)\n", 1),
    ("pdl8 fresh inputs",
     "            if not _pdl8h_guard(st, T, layer, off_max, off_bad, on_max, on_bad):\n                st[\"pdl8_on\"][T] = False\n",
     "            if not _pdl8h_guard(st, T, layer, off_max, off_bad, on_max, on_bad):\n                st[\"pdl8_on\"][T] = False\n"
     "            elif not _fresh_guard(st, T, layer, g_off, o_off, g_on, o_on, data, gen, cross=True):\n"
     "                st[\"pdl8_on\"][T] = False\n", 1),
    ("pdl8h budget",
     '                raise RuntimeError(f"PDL8H prepare budget {PDL8D_BUDGET_S:.0f} s reached before T=8 {layer}")\n',
     '                raise RuntimeError(f"PDL8H prepare budget {PDL8D_BUDGET_S:.0f} s reached before T=8 {layer}")\n'
     '            if _prep_left() < PREP_STAGE_MIN_S:\n'
     '                raise RuntimeError(f"prepare budget: {_prep_left():.0f} s left before T=8 {layer}")\n', 1),
    ("pdl8h fresh inputs",
     "            ok = _pdl8h_guard(st, 8, layer, off_max, off_bad, on_max, on_bad)\n",
     "            ok = _pdl8h_guard(st, 8, layer, off_max, off_bad, on_max, on_bad)\n"
     "            ok = ok and _fresh_guard(st, 8, layer, g_off, o_off, g_on, o_on, data, gen, cross=True)\n", 1),
    ("mpdl budget",
     '            print(f"MPDL_GUARD T={T}: PDL8 off at this T, edge off", flush=True)\n            continue\n',
     '            print(f"MPDL_GUARD T={T}: PDL8 off at this T, edge off", flush=True)\n            continue\n'
     '        if _prep_left() < PREP_STAGE_MIN_S:\n'
     '            print(f"MPDL_GUARD T={T}: prepare budget, {_prep_left():.0f} s left, edge off", flush=True)\n'
     '            st["moe_pdl"][T] = False\n'
     '            continue\n', 1),
    ("fresh-input helpers", "def _mpdl_layer(st, T, layer, gen, dev):\n", FRESH_HELPERS, 1),
    ("bfq fresh inputs",
     "        ok = _pdl8h_guard(st, T, layer, off_max, off_bad, on_max, on_bad)\n",
     "        ok = _pdl8h_guard(st, T, layer, off_max, off_bad, on_max, on_bad)\n"
     "        ok = ok and _fresh_guard(st, T, layer, g_off, o_off, g_on, o_on, data, gen, cross=False, tag=\"BFQ\")\n", 1),
    ("outermost prepare: budget start, late guard check",
     '_bfq_prior_prepare = prepare\n'
     'def prepare(weights):\n'
     '    st = _bfq_prior_prepare(weights)\n'
     '    st["bfq_on"] = False\n'
     '    try:\n'
     '        _BFQ["plan"], _BFQ["run"] = _bfq_selftest(st, weights["qkv_gate"].contiguous())\n'
     '        st["bfq_on"] = _bfq_guard(st)\n'
     '    except Exception as e:\n'
     '        print("BFQ off:", repr(e)[:400], flush=True)\n'
     '        st["bfq_on"] = False\n'
     '        torch.cuda.synchronize()\n'
     '    print(f"BFQ enabled {st[\'bfq_on\']}", flush=True)\n'
     '    return st\n',
     '_bfq_prior_prepare = prepare\n'
     'def prepare(weights):\n'
     '    global _PREP_T0\n'
     '    import time\n'
     '    _PREP_T0 = time.monotonic()\n'
     '    st = _bfq_prior_prepare(weights)\n'
     '    st["bfq_on"] = False\n'
     '    if _prep_left() < PREP_BFQ_MIN_S:\n'
     '        print(f"BFQ off: prepare budget, {_prep_left():.0f} s left", flush=True)\n'
     '    else:\n'
     '        try:\n'
     '            _BFQ["plan"], _BFQ["run"] = _bfq_selftest(st, weights["qkv_gate"].contiguous())\n'
     '            st["bfq_on"] = _bfq_guard(st)\n'
     '        except Exception as e:\n'
     '            print("BFQ off:", repr(e)[:400], flush=True)\n'
     '            st["bfq_on"] = False\n'
     '            torch.cuda.synchronize()\n'
     '    print(f"BFQ enabled {st[\'bfq_on\']}", flush=True)\n'
     '    _check_guards()\n'
     '    print(f"prepare done in {PREP_BUDGET_S - _prep_left():.1f} s", flush=True)\n'
     '    return st\n', 1),
    ("DE patches counted",
     " source=bfp_SK2_SRC.replace('@@SK2_DISPATCH@@',dispatch).replace('@@SK2F_DISPATCH@@','')\n"
     " source=source.replace('(NSTAGE >= 4 ? 4 : NSTAGE)','4').replace('NW + (COMP ? NESC : 0)','NW')\n"
     " source=source.replace('  static_assert(NESC <= NSTAGE, \"sk2: escape warps need NESC <= NSTAGE\");','')\n",
     " source=_patch(_patch(bfp_SK2_SRC,'@@SK2_DISPATCH@@',dispatch),'@@SK2F_DISPATCH@@','')\n"
     " source=_patch(_patch(source,'(NSTAGE >= 4 ? 4 : NSTAGE)','4',3),'NW + (COMP ? NESC : 0)','NW',4)\n"
     " source=_patch(source,'  static_assert(NESC <= NSTAGE, \"sk2: escape warps need NESC <= NSTAGE\");','')\n", 1),
    ("GLE config check",
     '    assert (layer, R, NWR * WSK, CL) == ("o", 32, 4, 1), cfg\n',
     '    if (layer, R, NWR * WSK, CL) != ("o", 32, 4, 1):\n        raise RuntimeError(f"gle: unexpected o config {cfg}")\n', 1),
    ("GLE patch 1 counted (open)", '    s = s.replace("#define SK2_DIAG 0", "#define SK2_DIAG 0\\n"\n',
     '    s = _patch(s, "#define SK2_DIAG 0", "#define SK2_DIAG 0\\n"\n', 1),
    ("GLE patch 1 counted (close)", '"static thread_local GleEpi g_gle;", 1)\n', '"static thread_local GleEpi g_gle;")\n', 1),
    ("GLE patch 2 counted",
     '    s = s.replace("           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {",\n'
     '                  "           float* __restrict__ part, int* __restrict__ rcnt, int pdl, const GleEpi ep) {", 1)\n',
     '    s = _patch(s, "           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {",\n'
     '               "           float* __restrict__ part, int* __restrict__ rcnt, int pdl, const GleEpi ep) {")\n', 1),
    ("GLE pre anchor", "    assert s.count(anchor) == 1\n    s = s.replace(anchor, pre + anchor, 1)\n",
     "    s = _patch(s, anchor, pre + anchor)\n", 1),
    ("GLE epi anchor", "    assert s.count(anchor) == 1\n    s = s.replace(anchor, epi, 1)\n", "    s = _patch(s, anchor, epi)\n", 1),
    ("GLE router edits", "        assert s.count(old) == 1, old[:60]\n        s = s.replace(old, new, 1)\n",
     "        s = _patch(s, old, new)\n", 1),
    ("GLE ilv edits", "        assert s.count(old) == 1, old[:70]\n        s = s.replace(old, new, 1)\n",
     "        s = _patch(s, old, new)\n", 1),
    ("GLE launch arg 1", '    s = s.replace("wsr, outr, part, rcnt, pdl));", "wsr, outr, part, rcnt, pdl, g_gle));", 1)\n',
     '    s = _patch(s, "wsr, outr, part, rcnt, pdl));", "wsr, outr, part, rcnt, pdl, g_gle));")\n', 1),
    ("GLE launch arg 2", '    s = s.replace("wsr, outr, part, rcnt, 0);", "wsr, outr, part, rcnt, 0, g_gle);", 1)\n',
     '    s = _patch(s, "wsr, outr, part, rcnt, 0);", "wsr, outr, part, rcnt, 0, g_gle);")\n', 1),
    ("GLE host anchor", "    assert s.count(host) == 1\n",
     '    if s.count(host) != 1:\n        raise RuntimeError("gle: sk2 host signature anchor")\n', 1),
    ("GLE dispatch placeholders",
     '    s = s.replace("@@SK2_DISPATCH@@", disp).replace("@@SK2F_DISPATCH@@", "\\n  ".join(flines))\n',
     '    s = _patch(_patch(s, "@@SK2_DISPATCH@@", disp), "@@SK2F_DISPATCH@@", "\\n  ".join(flines))\n', 1),
    ("fused pooler precondition",
     '    assert m <= 32 and state.get("sm90") and "pooler" in state["sk2"] and "router" in state["sk2"]\n',
     '    if not (m <= 32 and state.get("sm90") and "pooler" in state["sk2"] and "router" in state["sk2"]):\n'
     '        raise RuntimeError("fused pooler/router needs m <= 32 on sm_90 with both sk2 tables")\n', 2),
    ("GMB_TRIG constraint", "\nGMB_TRIG = 'entry'\n",
     "\n# Safe only while every PDL kernel after the b1 boundary reads h / hr after its griddepcontrol.wait\n"
     "# (edge proof: docs/decode-layer-final-review.md).\nGMB_TRIG = 'entry'\n", 1),
]


def remove_df_patch(text):
    start = text.index("\ndef _df_patch(c):\n")
    tail = ("_MA_ALT_SRCS['moe_helper0df'] = _MA_ALT_SRCS['moe_helper0']\n"
            "MOE_MID_DEFINES = tuple(\"@moe_helper0df\" if d == \"@moe_helper0\" else d for d in MOE_MID_DEFINES)\n")
    if text.count(tail) != 1 or text.count("def _df_patch(") != 1:
        raise SystemExit("df patch block not found exactly once")
    end = text.index(tail, start) + len(tail)
    if text[start:end].count("\ndef ") != 1:
        raise SystemExit("df patch block spans another function")
    text = text[:start] + "\n" + text[end:]
    if "_df_patch" in text or "moe_helper0df" in text:
        raise SystemExit("df patch still referenced")
    return text


def build():
    import hashlib
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != JUDGED_SHA256:
        raise SystemExit(f"{SRC} is not the judged eat_b4 source")
    before = raw.decode()
    text = normalize_whitespace(before)
    needles, hay = check_anchors_survive(before, text)
    for label, old, new, n in EDITS:
        c = text.count(old)
        if c != n:
            raise SystemExit(f"edit {label!r}: anchor found {c} times, expected {n}")
        text = text.replace(old, new)
    text = remove_df_patch(text)
    ast.parse(text)
    compile(text, str(DST), "exec")
    if "\n# okc: problem=muse_decode_layer\n" not in text:
        raise SystemExit("problem header lost")
    size = len(text.encode())
    if size >= SIZE_LIMIT:
        raise SystemExit(f"{size} bytes, limit {SIZE_LIMIT}")
    defs = lambda t: {n.name for n in ast.parse(t).body if isinstance(n, ast.FunctionDef)}
    added, removed = defs(text) - defs(before), defs(before) - defs(text)
    if added != {"_prep_left", "_patch", "_fresh_rounds", "_fresh_guard"} or removed != {"_df_patch"}:
        raise SystemExit(f"unexpected top-level functions: added {added}, removed {removed}")
    left = [m.start() for m in re.finditer(r"[ \t]+\n", text)]
    if left:
        raise SystemExit(f"trailing whitespace remains at {len(left)} places")
    return text, dict(edits=len(EDITS), anchors_checked=needles, haystacks=hay, bytes=size,
                      lines=text.count("\n") + 1, added=sorted(added), removed=sorted(removed))


def main():
    text, info = build()
    if "--check" in sys.argv:
        if DST.read_text() != text:
            raise SystemExit(f"{DST} is stale; rerun without --check")
        print("up to date", info)
        return
    DST.write_text(text)
    print("wrote", DST.relative_to(ROOT), info)


if __name__ == "__main__":
    main()
