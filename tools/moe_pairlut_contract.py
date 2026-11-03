"""Complete endpoint gate including unchanged-path drift."""
import math


def pairlut_decision(rows,result):
    drift=[]
    for row in rows:
        if row[1]<32:
            limit=max(math.log(1.01),2*max(row[6]))
            if any(abs(math.log(r))>limit for r in row[5]):drift.append(row[:3])
    return dict(passed=result['component_pass'] and not drift,component_pass=result['component_pass'],unchanged_drift=drift)
