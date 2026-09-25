"""String / set similarity kernels (numba, from scratch). Strings are uint8 slices buf[s:e]."""
import numpy as np
from numba import njit

_ONE = np.uint64(1)


@njit(cache=True)
def jaro_winkler(b1, s1, e1, b2, s2, e2):
    l1 = e1 - s1
    l2 = e2 - s2
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    rng = max(l1, l2) // 2 - 1
    if rng < 0:
        rng = 0
    m1 = np.zeros(l1, np.bool_)
    m2 = np.zeros(l2, np.bool_)
    matches = 0
    for i in range(l1):
        lo = max(0, i - rng)
        hi = min(i + rng + 1, l2)
        c = b1[s1 + i]
        for j in range(lo, hi):
            if not m2[j] and b2[s2 + j] == c:
                m1[i] = True
                m2[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    trans = 0
    k = 0
    for i in range(l1):
        if m1[i]:
            while not m2[k]:
                k += 1
            if b1[s1 + i] != b2[s2 + k]:
                trans += 1
            k += 1
    m = float(matches)
    jaro = (m / l1 + m / l2 + (m - trans / 2.0) / m) / 3.0
    pre = 0
    for i in range(min(4, l1, l2)):
        if b1[s1 + i] == b2[s2 + i]:
            pre += 1
        else:
            break
    return jaro + pre * 0.1 * (1.0 - jaro)


@njit(cache=True)
def _lev_myers(bp, sp, m, bt, st, n):
    """Myers/Hyyro bit-parallel Levenshtein; pattern length m <= 64."""
    peq = np.zeros(256, np.uint64)
    for i in range(m):
        peq[bp[sp + i]] |= _ONE << np.uint64(i)
    pv = ~np.uint64(0)
    mv = np.uint64(0)
    score = m
    last = _ONE << np.uint64(m - 1)
    for j in range(n):
        eq = peq[bt[st + j]]
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | ~(xh | pv)
        mh = pv & xh
        if ph & last:
            score += 1
        elif mh & last:
            score -= 1
        ph = (ph << _ONE) | _ONE
        mh = mh << _ONE
        pv = mh | ~(xv | ph)
        mv = ph & xv
    return score


@njit(cache=True)
def _lev_dp(b1, s1, l1, b2, s2, l2):
    prev = np.arange(l2 + 1)
    cur = np.zeros(l2 + 1, np.int64)
    for i in range(1, l1 + 1):
        cur[0] = i
        c = b1[s1 + i - 1]
        for j in range(1, l2 + 1):
            cost = 0 if c == b2[s2 + j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[l2]


@njit(cache=True)
def lev_ratio(b1, s1, e1, b2, s2, e2):
    l1 = e1 - s1
    l2 = e2 - s2
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    if l1 <= l2 and l1 <= 64:
        d = _lev_myers(b1, s1, l1, b2, s2, l2)
    elif l2 < l1 and l2 <= 64:
        d = _lev_myers(b2, s2, l2, b1, s1, l1)
    else:
        d = _lev_dp(b1, s1, l1, b2, s2, l2)
    return 1.0 - d / max(l1, l2)


@njit(cache=True)
def prefix_ratio(b1, s1, e1, b2, s2, e2):
    l1 = e1 - s1
    l2 = e2 - s2
    if l1 == 0 or l2 == 0:
        return 0.0
    k = 0
    for i in range(min(l1, l2)):
        if b1[s1 + i] != b2[s2 + i]:
            break
        k += 1
    return k / min(l1, l2)


@njit(cache=True)
def _trigrams(b, s, e):
    L = e - s
    out = np.empty(max(L, 0), np.int64)
    for j in range(L):
        c0 = 94 if j == 0 else b[s + j - 1]
        c1 = b[s + j]
        c2 = 36 if j == L - 1 else b[s + j + 1]
        out[j] = (np.int64(c0) << 16) | (np.int64(c1) << 8) | np.int64(c2)
    return np.sort(out)


@njit(cache=True)
def tri_dice(b1, s1, e1, b2, s2, e2):
    t1 = _trigrams(b1, s1, e1)
    t2 = _trigrams(b2, s2, e2)
    if len(t1) == 0 or len(t2) == 0:
        return 0.0
    i = 0
    j = 0
    inter = 0
    while i < len(t1) and j < len(t2):
        if t1[i] == t2[j]:
            inter += 1
            i += 1
            j += 1
        elif t1[i] < t2[j]:
            i += 1
        else:
            j += 1
    return 2.0 * inter / (len(t1) + len(t2))


@njit(cache=True)
def set_stats(d, p, a, b, idf):
    """Sorted-unique id sets of records a and b.
    -> (n_shared, n_a, n_b, idf2_shared, idf2_a, idf2_b, max_shared_idf)."""
    i, ie = p[a], p[a + 1]
    j, je = p[b], p[b + 1]
    na = ie - i
    nb = je - j
    sa = 0.0
    sb = 0.0
    for k in range(i, ie):
        sa += idf[d[k]] ** 2
    for k in range(j, je):
        sb += idf[d[k]] ** 2
    shared = 0
    ss = 0.0
    mx = 0.0
    while i < ie and j < je:
        x = d[i]
        y = d[j]
        if x == y:
            shared += 1
            w = idf[x]
            ss += w * w
            if w > mx:
                mx = w
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return shared, na, nb, ss, sa, sb, mx


@njit(cache=True)
def plain_jaccard(d, p, a, b):
    i, ie = p[a], p[a + 1]
    j, je = p[b], p[b + 1]
    n = (ie - i) + (je - j)
    if n == 0:
        return np.nan
    shared = 0
    while i < ie and j < je:
        if d[i] == d[j]:
            shared += 1
            i += 1
            j += 1
        elif d[i] < d[j]:
            i += 1
        else:
            j += 1
    return shared / (n - shared)


@njit(cache=True)
def _ndigits(x):
    k = 1
    while x >= 10:
        x //= 10
        k += 1
    return k


@njit(cache=True)
def num_stats(d, p, a, b):
    """House-number agreement -> (shared, n_a, n_b, fuzzy) where fuzzy means one unmatched
    number is a prefix/suffix of another (252 vs 52, 1681 vs 16)."""
    i0, ie = p[a], p[a + 1]
    j0, je = p[b], p[b + 1]
    shared = 0
    i = i0
    j = j0
    while i < ie and j < je:
        if d[i] == d[j]:
            shared += 1
            i += 1
            j += 1
        elif d[i] < d[j]:
            i += 1
        else:
            j += 1
    fuzzy = 0
    for i in range(i0, ie):
        x = d[i]
        for j in range(j0, je):
            y = d[j]
            if x == y or x < 10 or y < 10:
                continue
            big = x if x > y else y
            small = y if x > y else x
            dd = _ndigits(big) - _ndigits(small)
            if big % (10 ** _ndigits(small)) == small or big // (10 ** dd) == small:
                fuzzy = 1
    return shared, ie - i0, je - j0, fuzzy
