import numpy as np


def impl(
    actual_count,
    pq_distance,
    grouped_extreme_distance,
    pq_ivf,
    pq_index,
    order='ASC',
    k=None,
    group_size=None
):
    k = int(k)
    gs = int(group_size)
    n = int(np.asarray(actual_count))
    if k <= 0:
        raise ValueError('k should be bigger than zero, got %d' % k)
    if gs <= 0:
        raise ValueError('group_size should be bigger than zero, got %d' % gs)
    if n < 0 or n % gs != 0:
        raise ValueError('actual_count should be non-negative and multiple of group_size, got %d' % n)
    if k > n:
        raise ValueError('k(%d) should not be greater than actual_count(%d)' % (
            k,
            n
        ))
    des = str(order) != 'ASC'
    pq_arr = np.asarray(pq_distance)
    ivf_arr = np.asarray(pq_ivf)
    idx_arr = np.asarray(pq_index)
    vals = pq_arr[:n]
    key = (-vals).astype(np.float32) if des else vals.astype(np.float32)
    pos = np.argsort(key, kind='stable')[:k]
    topk_distance = vals[pos]
    topk_ivf = ivf_arr[:n][pos].astype(np.int32)
    topk_index = idx_arr[:n][pos].astype(np.int32)
    return (topk_distance, topk_ivf, topk_index)


def _kernel_reference(pq, ge, k, gs, asc):
    n = len(pq)
    pq = [float(v) for v in pq]
    ge = [float(v) for v in ge]
    extreme_size = n // gs
    size = k if k < extreme_size else extreme_size
    is_min_heap = not asc

    def fixdown(a, index, nn):
        i = index
        temp = a[i]
        j = (i << 1) + 1
        while j < nn:
            if is_min_heap:
                if j + 1 < nn and a[j][0] > a[j + 1][0]:
                    j += 1
                if a[j][0] >= temp[0]:
                    break
            else:
                if j + 1 < nn and a[j][0] < a[j + 1][0]:
                    j += 1
                if a[j][0] <= temp[0]:
                    break
            a[i] = a[j]
            i = j
            j = (i << 1) + 1
        a[i] = temp

    def make_heap(a, nn):
        for i in range((nn >> 1) - 1, -1, -1):
            fixdown(a, i, nn)

    def pop_heap(a, nn):
        res = a[0]
        a[0] = a[nn - 1]
        fixdown(a, 0, nn - 1)
        return res

    def sort_heap(a, nn):
        for i in range(nn - 1, -1, -1):
            temp = a[0]
            a[0] = a[i]
            a[i] = temp
            fixdown(a, 0, i)
    grp_extreme = []
    gi = 0
    cnt = 0
    while gi < extreme_size and cnt < size:
        grp_extreme.append((ge[gi], 0, gi))
        gi += 1
        cnt += 1
    make_heap(grp_extreme, size)
    while gi < extreme_size:
        temp = ge[gi]
        if is_min_heap:
            if grp_extreme[0][0] > temp:
                gi += 1
                continue
        elif grp_extreme[0][0] < temp:
            gi += 1
            continue
        grp_extreme[0] = (temp, 0, gi)
        fixdown(grp_extreme, 0, size)
        gi += 1
    sort_heap(grp_extreme, size)
    topk = []
    cnt = 0
    cntk = 0
    while cntk < len(grp_extreme):
        (grp, base) = (grp_extreme[cntk][1], grp_extreme[cntk][2] * gs)
        for index in range(gs):
            if cnt == k:
                break
            topk.append((pq[base + index], grp, base + index))
            cnt += 1
        if cnt == k:
            break
        cntk += 1
    assert cnt == k
    cntk = k // gs
    make_heap(topk, k)
    size2 = k if k < extreme_size else extreme_size
    index = k % gs
    while cntk < size2:
        if is_min_heap:
            if grp_extreme[cntk][0] <= topk[0][0]:
                cntk += 1
                index = 0
                continue
        elif grp_extreme[cntk][0] >= topk[0][0]:
            cntk += 1
            index = 0
            continue
        (grp, base) = (grp_extreme[cntk][1], grp_extreme[cntk][2] * gs)
        while index < gs:
            itemval = pq[base + index]
            if is_min_heap:
                if itemval <= topk[0][0]:
                    index += 1
                    continue
            elif itemval >= topk[0][0]:
                index += 1
                continue
            topk[0] = (itemval, grp, base + index)
            fixdown(topk, 0, k)
            index += 1
        cntk += 1
        index = 0
    out = []
    for i in range(k, 0, -1):
        out.append(pop_heap(topk, i))
    out.reverse()
    return [o[2] for o in out]


def _count_topk_ties(vals, k, des):
    sv = np.sort(np.asarray(vals, dtype=np.float64))
    region = (sv[-k:] if des else sv[:k]).tolist()
    bnd = None
    if len(sv) > k:
        bnd = sv[-k - 1] if des else sv[k]
    tie = 0
    run = 1
    for i in range(1, len(region)):
        if region[i] == region[i - 1]:
            run += 1
        else:
            if run > 1:
                tie += run
            run = 1
    if run > 1:
        tie += run
    if bnd is not None and bnd in region:
        tie += region.count(bnd)
    return tie
