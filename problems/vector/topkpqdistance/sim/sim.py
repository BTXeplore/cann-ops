"""Python mirror of the Ascend C algorithm (heap semantics identical) to validate logic."""
import numpy as np
import sys
sys.path.insert(0, '.')
from topkpq import reference


def is_worse(av, ai, bv, bi, desc):
    if desc:
        if av < bv:
            return True
        if av > bv:
            return False
        return ai > bi
    if av > bv:
        return True
    if av < bv:
        return False
    return ai > bi


def sift_down(hv, hi, start, size, desc):
    tv, ti = hv[start], hi[start]
    parent = start
    child = (parent << 1) + 1
    while child < size:
        right = child + 1
        if right < size and is_worse(hv[right], hi[right], hv[child], hi[child], desc):
            child = right
        if not is_worse(hv[child], hi[child], tv, ti, desc):
            break
        hv[parent], hi[parent] = hv[child], hi[child]
        parent = child
        child = (parent << 1) + 1
    hv[parent], hi[parent] = tv, ti


def plan(n, k, cores):
    blockNum = min(cores, 64, n)
    blockNum = max(blockNum, 1)
    chunk = (n + blockNum - 1) // blockNum
    chunk = ((chunk + 7) // 8) * 8
    blockNum = (n + chunk - 1) // chunk
    kAligned = ((k + 7) // 8) * 8
    return blockNum, chunk, kAligned


def stage1(vals, n, k, chunk, blockNum, desc):
    out = []
    kAligned = ((k + 7) // 8) * 8
    for b in range(blockNum):
        lo = b * chunk
        if lo >= n:
            out.append(([], []))
            continue
        hi = min(lo + chunk, n)
        cnt = hi - lo
        m = min(k, cnt)
        hv = [0.0] * m
        hi_ = [0] * m
        seg = vals[lo:hi]
        for i in range(m):
            hv[i] = float(seg[i])
            hi_[i] = lo + i
        for q in range((m >> 1) - 1, -1, -1):
            sift_down(hv, hi_, q, m, desc)
        rootv, rooti = hv[0], hi_[0]
        for i in range(m, cnt):
            v = float(seg[i])
            idx = lo + i
            if is_worse(rootv, rooti, v, idx, desc):
                hv[0], hi_[0] = v, idx
                sift_down(hv, hi_, 0, m, desc)
                rootv, rooti = hv[0], hi_[0]
        out.append((hv, hi_))
    return out


def stage2(cands, n, k, chunk, blockNum, desc):
    m = k
    hv = [0.0] * m
    hi_ = [0] * m
    filled = 0
    for b in range(blockNum):
        lo = b * chunk
        if lo >= n:
            break
        cnt = min(chunk, n - lo)
        cb = min(k, cnt)
        cv, ci = cands[b]
        for i in range(cb):
            v, idx = cv[i], ci[i]
            if filled < m:
                hv[filled], hi_[filled] = v, idx
                filled += 1
                if filled == m:
                    for q in range((m >> 1) - 1, -1, -1):
                        sift_down(hv, hi_, q, m, desc)
            elif is_worse(hv[0], hi_[0], v, idx, desc):
                hv[0], hi_[0] = v, idx
                sift_down(hv, hi_, 0, m, desc)
    assert filled == m, (filled, m)
    for i in range(m - 1):
        rest = m - 1 - i
        topV, topI = hv[0], hi_[0]
        lastV, lastI = hv[rest], hi_[rest]
        hv[rest], hi_[rest] = topV, topI
        hv[0], hi_[0] = lastV, lastI
        sift_down(hv, hi_, 0, rest, desc)
    return hv, hi_


def run(actual_count, pq_distance, pq_ivf, pq_index, order, k, group_size, cores):
    n = int(actual_count)
    desc = 1 if str(order) != 'ASC' else 0
    kk = min(k, n)
    blockNum, chunk, kAligned = plan(n, k, cores)
    vals = np.asarray(pq_distance)[:n].astype(np.float64)
    cands = stage1(vals, n, kk, chunk, blockNum, desc)
    v, idx = stage2(cands, n, kk, chunk, blockNum, desc)
    return np.asarray(v), np.asarray(idx, dtype=np.int64)


def main():
    rng = np.random.default_rng(7)
    fails = 0
    cases = []
    # 上游 UT 用例
    cases.append((12, [1, 2, 3, 4, 12, 13, 5, 1, 4, 2, 3, 2], 'DES', 5, 2))
    cases.append((8, [5, 1, 9, 3, 7, 2, 8, 6], 'ASC', 4, 2))
    for t in range(300):
        n = int(rng.integers(1, 60))
        gs = int(rng.choice([1, 2, 3, 5, n]))
        if n % gs != 0:
            gs = 1
        k = int(rng.integers(1, n + 1))
        order = 'ASC' if t % 2 == 0 else 'DES'
        vals = rng.uniform(1, 1000000, n) if order == 'ASC' else -rng.uniform(1, 1000000, n)
        cases.append((n, vals.tolist(), order, k, gs))
    for (n, vals, order, k, gs) in cases:
        vals = np.asarray(vals, dtype=np.float32)
        ivf = rng.integers(0, 1000, n).astype(np.int32)
        idx = rng.integers(0, 10 ** 8, n).astype(np.int32)
        exp_v, exp_ivf, exp_idx = reference.impl(
            n, vals, np.zeros(max(1, n // gs), np.float32), ivf, idx,
            order=order, k=k, group_size=gs)
        for cores in (1, 4, 40):
            got_v, got_pos = run(n, vals, ivf, idx, order, k, gs, cores)
            ok = (np.array_equal(got_v.astype(np.float32), exp_v.astype(np.float32))
                  and np.array_equal(ivf[got_pos], exp_ivf.astype(np.int32))
                  and np.array_equal(idx[got_pos], exp_idx.astype(np.int32))
                  and np.array_equal(got_pos, np.argsort(
                      (-vals if order != 'ASC' else vals).astype(np.float32), kind='stable')[:k]))
            if not ok:
                fails += 1
                print('FAIL', n, order, k, gs, cores)
                print(' exp', exp_v[:8], exp_ivf[:8])
                print(' got', got_v[:8], ivf[got_pos][:8])
                break
    print('cases', len(cases), 'fails', fails)


if __name__ == '__main__':
    main()
