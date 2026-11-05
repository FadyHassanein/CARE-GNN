"""The precomputed Jaccard table must reproduce the set-based formula exactly.

MultiViewDistance used to compute |N(v) & N(u)| / |N(v) | N(u)| per pair with
Python sets. That is O(deg^2) and fell over on Amazon (mean degree 598), so the
distances are now precomputed per edge via a sparse product. This checks the
replacement is value-identical, not merely close.

    python test_jaccard_table.py
"""
import pickle
import random

import numpy as np

from capn import MultiViewDistance


def reference(adj, center, neigh):
    """The original set-based structural distance."""
    c = adj.get(center, set())
    n = adj.get(neigh, set())
    return 1.0 - len(c & n) / max(len(c | n), 1)


def check(name, path, n_centres=60, seed=0):
    adj = pickle.load(open(path, 'rb'))
    mvd = MultiViewDistance(1, gamma_init=0.7, adj_lists=[adj])
    rng = random.Random(seed)
    centres = rng.sample(sorted(int(k) for k in adj), min(n_centres, len(adj)))

    pairs = 0
    worst = 0.0
    for v in centres:
        neighs = sorted(int(u) for u in adj.get(v, set()))
        if not neighs:
            continue
        # a few non-neighbours too, to exercise the fallback path
        probe = neighs + rng.sample(sorted(int(k) for k in adj), 3)
        got = mvd._struct_dists(v, probe, 0)
        want = np.array([reference(adj, v, u) for u in probe])
        assert got.shape == want.shape, f'{name}: shape {got.shape} != {want.shape}'
        worst = max(worst, float(np.abs(got - want).max()))
        pairs += len(probe)

    assert worst == 0.0, f'{name}: tables differ from set formula by {worst}'
    print(f'  {name}: {pairs} pairs over {len(centres)} centres -- exact match')


if __name__ == '__main__':
    print('Jaccard table vs. set-based reference:')
    for name, path in [('yelp rur', 'data/yelp_rur_adjlists.pickle'),
                       ('yelp rtr', 'data/yelp_rtr_adjlists.pickle'),
                       ('amazon upu', 'data/amz_upu_adjlists.pickle'),
                       ('amazon usu', 'data/amz_usu_adjlists.pickle')]:
        check(name, path)
    print('OK')
