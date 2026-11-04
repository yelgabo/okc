import json, sys, statistics, math
import numpy as np
d = json.load(open(sys.argv[1]))
for row in d["rows"]:
    c = row["case"]; label = f'{c["load"]} {c["layer"]}'
    inputs = [i for i in row["inputs"] if all(not math.isnan(y) for y in i["yA"]) and all(not math.isnan(y) for y in i["yB"])]
    ni = len(inputs); n = len(inputs[0]["yA"])
    # bend_rel per CTA = yA + yB (first issue nearly uniform); fit on per-NTT counts A and B with a per-input intercept
    X, Y, idx = [], [], []
    for k, inp in enumerate(inputs):
        for i in range(n):
            X.append(inp["rowsA"][i] + inp["rowsB"][i] + [1.0 if kk == k else 0.0 for kk in range(ni)])
            Y.append(inp["yA"][i] + inp["yB"][i]); idx.append(i)
    X = np.array(X, float); Y = np.array(Y)
    used = [j for j in range(16) if X[:, j].sum() > 0] + list(range(16, 16 + ni))
    cf, *_ = np.linalg.lstsq(X[:, used], Y, rcond=None)
    fitted = X[:, used] @ cf; res = Y - fitted
    # per input: sd of Y, sd of fitted (deterministic part), sd of residual; max - mean of each
    sy, sf, sr, my, mf, mr = [], [], [], [], [], []
    for k in range(ni):
        sel = slice(k * n, (k + 1) * n)
        y, f, r = Y[sel], fitted[sel], res[sel]
        sy.append(y.std()); sf.append(f.std()); sr.append(r.std())
        my.append(y.max() - y.mean()); mf.append(f.max() - f.mean()); mr.append(r.max() - r.mean())
    cA = [cf[used.index(j)] if j in used else math.nan for j in range(8)]
    cB = [cf[used.index(j)] if j in used else math.nan for j in range(8, 16)]
    print(f"{label}: n inputs {ni}; per-CTA end (yA+yB) sd median {statistics.median(sy):.2f}, fitted-part sd {statistics.median(sf):.2f}, residual sd {statistics.median(sr):.2f}")
    print(f"  max-mean: total {statistics.median(my):.2f}, fitted part {statistics.median(mf):.2f}, residual part {statistics.median(mr):.2f}")
    print("  coef A per NTT " + " ".join(f"{x:.3f}" for x in cA) + " | B " + " ".join(f"{x:.3f}" for x in cB))
    # per blockIdx mean residual over inputs: a deterministic position effect?
    rb = np.zeros(n);
    for k in range(ni):
        rb += res[k * n:(k + 1) * n]
    rb /= ni
    order = np.argsort(rb)
    print(f"  per-blockIdx mean residual: sd {rb.std():.2f}; most late idx " + " ".join(f"{i}:{rb[i]:+.2f}" for i in order[-8:]) + " ; most early " + " ".join(f"{i}:{rb[i]:+.2f}" for i in order[:8]))
    # correlation of residual between consecutive inputs by blockIdx
    cc = [np.corrcoef(res[k * n:(k + 1) * n], res[(k + 1) * n:(k + 2) * n])[0, 1] for k in range(ni - 1)]
    print(f"  residual corr between consecutive inputs by blockIdx: median {statistics.median(cc):.2f}")
    # how much of the spread is in the phase B part alone: fit yB on rowsB only
    XB = np.array([inp["rowsB"][i] + [1.0 if kk == k else 0.0 for kk in range(ni)] for k, inp in enumerate(inputs) for i in range(n)], float)
    YB = np.array([inp["yB"][i] for inp in inputs for i in range(n)])
    YA = np.array([inp["yA"][i] for inp in inputs for i in range(n)])
    usedB = [j for j in range(8) if XB[:, j].sum() > 0] + list(range(8, 8 + ni))
    cfb, *_ = np.linalg.lstsq(XB[:, usedB], YB, rcond=None); resB = YB - XB[:, usedB] @ cfb
    print(f"  phase B alone: yB sd {statistics.median([YB[k*n:(k+1)*n].std() for k in range(ni)]):.2f}, residual sd {statistics.median([resB[k*n:(k+1)*n].std() for k in range(ni)]):.2f}; corr(yA, yB) within input median {statistics.median([np.corrcoef(YA[k*n:(k+1)*n], YB[k*n:(k+1)*n])[0,1] for k in range(ni)]):.2f}")
