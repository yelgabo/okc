"""Offline test of _stable_pick (tools/dl_stable_tuner_patch.py) on synthetic round times: a true
tie stays on the preferred pick across many noise draws, a consistent win beyond the margin
switches, a win with three disagreeing rounds does not, a noisy preferred pick widens the
margin, and an absent preferred pick falls back to the lowest median.

    python3 tools/dl_stable_pick_test.py tools
"""
import random
import sys

sys.path.insert(0, sys.argv[1])
import dl_stable_tuner_patch as sp  # noqa: E402

ns = {}
exec(sp.HELPER, ns)
pick = ns["_stable_pick"]
R = ns["STABLE_ROUNDS"]
rng = random.Random(7)


def draw(mean, sd):
    return [mean + rng.gauss(0, sd) for _ in range(R)]


bad = 0
flips = argmin = 0
for _ in range(2000):
    rounds = {"a": draw(100, 0.4), "b": draw(100, 0.4)}
    flips += pick(rounds, "a")[0] != "a"
    med = {k: sorted(v)[R // 2] for k, v in rounds.items()}
    argmin += min(med, key=med.get) != "a"
ok = flips <= 100
bad += not ok
print(f"{'ok  ' if ok else 'FAIL'} true tie, noise 0.4 us: {flips}/2000 off the preferred pick "
      f"(lowest median of the same rounds: {argmin}/2000)")
wins = sum(pick({"a": draw(100, 0.3), "b": draw(98, 0.3)}, "a")[0] == "b" for _ in range(500))
ok = wins >= 475
bad += not ok
print(f"{'ok  ' if ok else 'FAIL'} 2 us real win, noise 0.3 us: switched {wins}/500")
base = [100.0] * R
other = [97.0] * (R - 3) + [101.0] * 3
got, note = pick({"a": base, "b": other}, "a")
ok = got == "a"
bad += not ok
print(f"{'ok  ' if ok else 'FAIL'} 3 us median win, 3 rounds disagree: {note}")
noisy = [100, 103, 97, 104, 96, 100, 102, 98, 100][:R]
got, note = pick({"a": noisy, "b": [x - 1.5 for x in noisy]}, "a")
ok = got == "a" and "margin 4.00" in note
bad += not ok
print(f"{'ok  ' if ok else 'FAIL'} 1.5 us win under a 2 x MAD = 4 us margin: {note}")
got, note = pick({"b": [10.0] * R, "c": [9.0] * R}, "a")
ok = got == "c" and "absent" in note
bad += not ok
print(f"{'ok  ' if ok else 'FAIL'} preferred absent: {note}")
print("all cases as expected" if not bad else f"{bad} cases failed")
sys.exit(1 if bad else 0)
