"""Small numba helpers shared by several stages."""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def csr_sort_unique(indptr, data):
    """Sort + deduplicate each CSR segment. Returns new (indptr, data)."""
    n = len(indptr) - 1
    out = np.empty_like(data)
    new_ptr = np.zeros(n + 1, np.int64)
    pos = 0
    for r in range(n):
        seg = np.sort(data[indptr[r]:indptr[r + 1]])
        for j in range(len(seg)):
            if j == 0 or seg[j] != seg[j - 1]:
                out[pos] = seg[j]
                pos += 1
        new_ptr[r + 1] = pos
    return new_ptr, out[:pos].copy()


def group_rank_desc(group, score):
    """1-based rank of each row's score within its group (descending). group need not be sorted."""
    order = np.lexsort((-score, group))
    g = group[order]
    idx = np.arange(len(g))
    start = np.r_[True, g[1:] != g[:-1]] if len(g) else np.zeros(0, bool)
    grp_start = np.maximum.accumulate(np.where(start, idx, 0)) if len(g) else idx
    rank = np.empty(len(g), np.int32)
    rank[order] = idx - grp_start + 1
    return rank


@njit(cache=True)
def group_top2(group, score, n_groups):
    """Per group: max, second max, sum, count(score > 0.5), size."""
    top1 = np.full(n_groups, -1.0)
    top2 = np.full(n_groups, -1.0)
    tot = np.zeros(n_groups)
    c05 = np.zeros(n_groups, np.int32)
    size = np.zeros(n_groups, np.int32)
    for i in range(len(group)):
        g = group[i]
        s = score[i]
        size[g] += 1
        tot[g] += s
        if s > 0.5:
            c05[g] += 1
        if s > top1[g]:
            top2[g] = top1[g]
            top1[g] = s
        elif s > top2[g]:
            top2[g] = s
    return top1, top2, tot, c05, size


@njit(parallel=True, cache=True)
def take_segments_len(indptr, idx):
    out = np.empty(len(idx), np.int64)
    for i in prange(len(idx)):
        out[i] = indptr[idx[i] + 1] - indptr[idx[i]]
    return out
