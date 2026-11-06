import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'topkpq'))
from sim import run


def check(n, k, gs, order, cores, seed=0):
    rng = np.random.default_rng(seed)
    vals = rng.uniform(1, 1_000_000.0, n) if order == 'ASC' else -rng.uniform(1, 1_000_000.0, n)
    vals = vals.astype(np.float32)
    ivf = rng.integers(0, 100000, n).astype(np.int32)
    idx = rng.integers(0, 10 ** 8, n).astype(np.int32)
    got_v, got_pos = run(n, vals, ivf, idx, order, k, gs, cores)
    exp_pos = np.argsort((-vals if order != 'ASC' else vals).astype(np.float32), kind='stable')[:k]
    ok = (np.array_equal(got_v.astype(np.float32), vals[exp_pos]) and np.array_equal(got_pos, exp_pos))
    print(('OK  ' if ok else 'FAIL'), 'n=%d k=%d gs=%d %s cores=%d' % (n, k, gs, order, cores))
    return ok


cases = []
for n in (1, 2, 3, 7, 8, 9, 15, 16, 17):
    for k in (1, n):
        cases.append((n, k, 1))
cases += [
    (1000, 500, 2), (1000, 1, 1), (1000, 1000, 10), (2048, 1000, 4),
    (100003, 1000, 1), (100003, 37, 7), (65536, 5000, 8), (99, 99, 3),
    (100000, 3, 1), (77777, 4999, 11),
]
fails = 0
for (n, k, gs) in cases:
    for order in ('ASC', 'DES'):
        for cores in (1, 40):
            if not check(n, k, gs, order, cores):
                fails += 1
print('fails', fails)
