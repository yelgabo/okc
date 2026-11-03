"""CPU model of the OKC_BCLM phase B claim protocol (tools/moe_tiny_claim.py), one launch at a time.

It mirrors the native control flow position by position: the static ranges (muldiv floor splits), thread 0's refills
with the claim issued one fill ahead, the end marker in S.bq, the WSBL kept positions, the compute warps' read of
S.bq one position ahead for the epilogue prefetch, and the helper's queue ending at S.bh = -1. CTAs advance in a
random interleaving, so claims land in a random order. It checks that every phase B unit is computed and finished
(by the helper or a WSBL warp) exactly once, that every read of S.bq / S.bh sees a value written for that position,
and that each WSBL warp prefetched the constants of the unit it finishes.

    python3 tools/moe_tiny_claim_model.py [--trials 3000]
"""
import random
import sys

NS, NRB = 3, 4


def split(a, c, G):
    return a * c // G


def launch(G, njS, K, k, rng):
    nA, nB = njS * 512, njS * 256
    lim, P0 = nB - G * (NS + 1), G * k
    P = 0 if lim <= 0 else min(P0, lim)
    nBs = nB - P
    dyn = P > 0
    ctr = [0]
    finished = []
    ctas = []
    for c in range(G):
        a0, a1 = split(nA, c, G), split(nA, c + 1, G)
        b0, b1 = split(nBs, c, G), split(nBs, c + 1, G)
        st = dict(c=c, a0=a0, nAc=a1 - a0, b0=b0, nU=b1 - b0, q=0, bq={}, bh={}, pend=None, qend=None, hend=False,
                  vp=[], prefetch={}, helper=[], done=False, phaseB=False, i=0)
        st['nS'] = st['nAc'] + st['nU']
        if dyn:
            assert st['nU'] >= NS + 1, (G, njS, c, st['nU'])
        ctas.append(st)

    def unit_at(st, q):
        """The unit a static position holds (phase, unit)."""
        if q < st['nAc']:
            return 'A', st['a0'] + q
        return 'B', st['b0'] + (q - st['nAc'])

    def step(st):
        q = st['q']
        if q < st['nAc']:  # phase A iteration (release loop): thread 0 refills static positions only
            qf = q + NS
            assert qf < st['nS'] or not dyn
            st['helper'].append(('A', st['a0'] + q))
            st['q'] += 1
            return
        i = q - st['nAc']
        if i < st['nU']:
            v = st['b0'] + i
        else:
            if not dyn:
                st['done'] = True
                return finish(st, i)
            assert q in st['bq'], ('read before fill', st['c'], q)
            v = st['bq'][q]
            if v < 0:
                st['done'] = True
                return finish(st, i)
        for w in range(K):
            if not dyn:
                if i == st['nU'] - 1 - w:
                    st['prefetch'][w] = v
            elif q + 1 >= st['nS']:
                assert q + 1 in st['bq'], ('prefetch read before fill', st['c'], q + 1)
                if st['bq'][q + 1] < 0:
                    vk = v if w == 0 else (st['vp'][-1] if st['vp'] else -1)
                    if vk >= 0:
                        st['prefetch'][w] = vk
        st['vp'].append(v)
        # thread 0 after the CTA barrier
        kept = (q + K >= st['qend']) if (dyn and st['qend'] is not None) else (False if dyn else i >= st['nU'] - K)
        qf = q + NS
        if qf < st['nS']:
            if dyn and qf + 1 == st['nS']:
                st['pend'] = ctr[0]; ctr[0] += 1
        elif dyn and (st['qend'] is None or qf < st['qend']):
            assert st['pend'] is not None, 'claim used before issue'
            u = st['pend']
            if u < P:
                st['bq'][qf] = nBs + u
                st['pend'] = ctr[0]; ctr[0] += 1
            else:
                st['bq'][qf] = -1
                st['qend'] = qf
        if not kept:
            st['bh'][q] = v
            st['helper'].append(('B', v))
        elif dyn and not st['hend']:
            st['bh'][q] = -1
            st['hend'] = True
        st['q'] += 1

    def finish(st, i):
        for w in range(K):
            if w < i:
                v = st['vp'][-1 - w]
                assert st['prefetch'].get(w) == v, ('prefetch mismatch', st['c'], w, st['prefetch'].get(w), v)
                finished.append(('B', v, 'warp'))
        if dyn:
            kept = st['vp'][-K:] if i >= K else st['vp']
            # the helper stops at its end marker, which thread 0 sent at the first kept position
            assert st['hend'], ('no helper end marker', st['c'])
            hb = [u for ph, u in st['helper'] if ph == 'B']
            assert not set(hb) & set(kept)
        else:
            nh = st['nS'] - min(K, st['nU'])
            assert len(st['helper']) == nh, (len(st['helper']), nh)
        for ph, u in st['helper']:
            finished.append((ph, u, 'helper'))

    live = list(ctas)
    while live:
        st = rng.choice(live)
        step(st)
        if st['done']:
            live.remove(st)
    a = sorted(u for ph, u, _ in finished if ph == 'A')
    b = sorted(u for ph, u, _ in finished if ph == 'B')
    assert a == list(range(nA)), 'phase A units'
    assert b == list(range(nB)), ('phase B units', len(b), nB, P)
    return P


def main():
    trials = int(sys.argv[sys.argv.index('--trials') + 1]) if '--trials' in sys.argv else 3000
    rng = random.Random(20261103)
    seen = {}
    for t in range(trials):
        G = rng.choice([132, 132, 132, 100, 117, 136])
        njS = rng.choice([1, 2, 3, 4, 6, 7, 8, 10, 16, 32])
        K = rng.choice([1, 2])
        k = rng.choice([1, 2, 3])
        P = launch(G, njS, K, k, rng)
        seen[(njS, P > 0)] = seen.get((njS, P > 0), 0) + 1
    print('ok', trials, 'launches; (jobs, pool on) counts:', dict(sorted(seen.items())))


if __name__ == '__main__':
    main()
