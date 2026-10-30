"""Per-kernel SASS identity between two tools/proj_sass.sh dumps, plus the ptxas resource lines
and C75xx (wgmma serialization) warnings of the kernels that are new or changed.

    python3 tools/proj_sass_diff.py BASE_PREFIX NEW_PREFIX

Instruction text is compared with addresses and encodings stripped; a kernel counts as
identical when its instruction sequence is. Exit status 1 if any kernel present in both
dumps differs.
"""

import re
import sys
from pathlib import Path

# The anonymous namespace's mangled name carries a hash of the translation unit.
ANON = re.compile(r"_GLOBAL__N__[0-9a-f]+_\d+_sk_cu_[0-9a-f]+_\d+")


def functions(path):
    out, name, body = {}, None, []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            if name:
                out[name] = body
            name, body = ANON.sub("ANON", m.group(1)), []
            continue
        if name is None:
            continue
        ins = re.sub(r"/\*[0-9a-fx]+\*/", "", line)
        ins = ANON.sub("ANON", re.sub(r"/\* 0x[0-9a-f]+ \*/", "", ins)).strip()
        if ins and not ins.startswith(".") and "........" not in ins:
            body.append(ins)
    if name:
        out[name] = body
    return out


def ptxas_info(path):
    """mangled name -> (registers line, spill line), and every C75xx warning line."""
    info, cur, warns = {}, None, []
    for line in Path(path).read_text().splitlines():
        m = re.search(r"Compiling entry function '(\S+)'", line)
        if m:
            cur = ANON.sub("ANON", m.group(1))
            info[cur] = ["", ""]
        elif cur and "registers" in line:
            info[cur][0] = line.split("info    :")[-1].strip()
        elif cur and "spill" in line:
            info[cur][1] = line.split("info    :")[-1].strip()
        if re.search(r"C75\d\d", line):
            warns.append(line.strip())
    return info, warns


def main():
    base, new = sys.argv[1], sys.argv[2]
    a, b = functions(base + ".sass"), functions(new + ".sass")
    same = [f for f in a if f in b and a[f] == b[f]]
    diff = [f for f in a if f in b and a[f] != b[f]]
    gone = [f for f in a if f not in b]
    added = [f for f in b if f not in a]
    print(f"base {len(a)} kernels, new {len(b)}: {len(same)} identical, {len(diff)} changed, "
          f"{len(added)} added, {len(gone)} removed")
    info, warns = ptxas_info(new + ".ptxas")
    for f in diff + added:
        regs, spill = info.get(f, ["?", "?"])
        tag = "CHANGED" if f in diff else "ADDED"
        print(f"{tag} {f}\n    {regs}\n    {spill}\n    {len(b[f])} instructions")
    for f in gone:
        print(f"REMOVED {f}")
    spills = [f for f, (r, s) in info.items() if s and not re.search(r"0 bytes spill stores, 0 bytes spill loads", s)]
    print(f"kernels with spills (new dump): {len(spills)}" + "".join(f"\n    {f}: {info[f][1]}" for f in spills))
    print(f"C75xx warnings (new dump): {len(warns)}" + "".join(f"\n    {w}" for w in warns))
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
