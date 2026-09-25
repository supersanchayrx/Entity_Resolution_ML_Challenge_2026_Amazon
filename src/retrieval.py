"""Stage `block`: candidate generation by IDF-weighted sparse retrieval (from scratch).

Every record becomes a sparse vector over hashed features:
  1 name tokens, 2 name character trigrams (typo/domain robust), 3 address tokens,
  4 house numbers, 5 number x address-token combos, 6 number x name-token combos,
  7 glued name-token spans ("legacy bright alphabet" -> legacybrightalphabet) for domain names.
Weights = type multiplier * IDF over the S2/S3 pool; features above a document-frequency cap
are dropped (stop-word pruning) and each S1 query keeps only its m heaviest features
(prefix filtering). Cosine scores come from chunked sparse matmuls, run per country.
Forward: top-k1 pool records per S1. Reverse: top rev_k S1s per pool record.
"""
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from numba import njit

from .nbutils import csr_sort_unique, group_rank_desc, group_top2

MASK40 = np.int64((1 << 40) - 1)
TYPES = ["", "ntok", "tri", "atok", "num", "numx", "nnum", "span"]
MAX_SPAN_TOK = 6


@njit(cache=True)
def _mix(x):
    h = np.uint64(x)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xC4CEB9FE1A85EC53)
    h ^= h >> np.uint64(33)
    return np.int64(h & np.uint64(0xFFFFFFFFFF))


@njit(cache=True)
def _span_hash(buf, starts, ends, i, j):
    """FNV-1a over tokens i..j (inclusive) glued together, 40-bit."""
    h = np.uint64(1469598103934665603)
    for t in range(i, j + 1):
        for k in range(starts[t], ends[t]):
            h ^= np.uint64(buf[k])
            h *= np.uint64(1099511628211)
    return np.int64(h & np.uint64(0xFFFFFFFFFF))


@njit(cache=True)
def build_feats(rows, nt_p, nt_d, nx_p, nx_b, at_p, at_d, nu_p, nu_d, nc_p, nc_b):
    n = len(rows)
    cnt = np.zeros(n, np.int64)
    for i in range(n):
        r = rows[i]
        nn = nt_p[r + 1] - nt_p[r]
        na = at_p[r + 1] - at_p[r]
        nnum = nu_p[r + 1] - nu_p[r]
        k3 = min(nnum, 3)
        T = 1
        for k in range(nc_p[r], nc_p[r + 1]):
            if nc_b[k] == 32:
                T += 1
        T = min(T, MAX_SPAN_TOK)
        cnt[i] = (nn + (nx_p[r + 1] - nx_p[r]) + na + nnum + k3 * min(na, 10) + k3 * min(nn, 6)
                  + T * (T + 1) // 2)
    ptr = np.zeros(n + 1, np.int64)
    for i in range(n):
        ptr[i + 1] = ptr[i] + cnt[i]
    out = np.empty(ptr[n], np.int64)
    for i in range(n):
        r = rows[i]
        k = ptr[i]
        for j in range(nt_p[r], nt_p[r + 1]):
            out[k] = (np.int64(1) << 40) | np.int64(nt_d[j])
            k += 1
        s, e = nx_p[r], nx_p[r + 1]
        L = e - s
        for j in range(L):
            c0 = 94 if j == 0 else nx_b[s + j - 1]
            c2 = 36 if j == L - 1 else nx_b[s + j + 1]
            out[k] = (np.int64(2) << 40) | (np.int64(c0) << 16) | (np.int64(nx_b[s + j]) << 8) | np.int64(c2)
            k += 1
        for j in range(at_p[r], at_p[r + 1]):
            out[k] = (np.int64(3) << 40) | np.int64(at_d[j])
            k += 1
        for j in range(nu_p[r], nu_p[r + 1]):
            out[k] = (np.int64(4) << 40) | (nu_d[j] & MASK40)
            k += 1
        nnum = min(nu_p[r + 1] - nu_p[r], 3)
        for q in range(nnum):
            num = nu_d[nu_p[r] + q]
            for j in range(at_p[r], min(at_p[r + 1], at_p[r] + 10)):
                out[k] = (np.int64(5) << 40) | _mix(num * 1000003 + np.int64(at_d[j]))
                k += 1
            for j in range(nt_p[r], min(nt_p[r + 1], nt_p[r] + 6)):
                out[k] = (np.int64(6) << 40) | _mix(num * 7919 + np.int64(nt_d[j]) * 1000003 + 17)
                k += 1
        # glued spans of consecutive core-name tokens (length >= 2, or the single token)
        starts = np.empty(MAX_SPAN_TOK, np.int64)
        ends = np.empty(MAX_SPAN_TOK, np.int64)
        T = 0
        cs = nc_p[r]
        for q in range(nc_p[r], nc_p[r + 1] + 1):
            if q == nc_p[r + 1] or nc_b[q] == 32:
                if q > cs and T < MAX_SPAN_TOK:
                    starts[T] = cs
                    ends[T] = q
                    T += 1
                cs = q + 1
        for i0 in range(T):
            for j0 in range(i0, T):
                if j0 > i0 or T == 1:
                    out[k] = (np.int64(7) << 40) | _span_hash(nc_b, starts, ends, i0, j0)
                    k += 1
        ptr_fill_end = ptr[i + 1]
        while k < ptr_fill_end:  # unused slots (pad with a harmless duplicate)
            out[k] = out[k - 1] if k > ptr[i] else (np.int64(7) << 40)
            k += 1
    return ptr, out


@njit(cache=True)
def _prefix_filter(ptr, idx, data, m):
    n = len(ptr) - 1
    cnt = np.zeros(n + 1, np.int64)
    for r in range(n):
        cnt[r + 1] = cnt[r] + min(m, ptr[r + 1] - ptr[r])
    oi = np.empty(cnt[n], idx.dtype)
    od = np.empty(cnt[n], data.dtype)
    for r in range(n):
        a, b = ptr[r], ptr[r + 1]
        seg = data[a:b]
        order = np.argsort(-seg)[:min(m, b - a)]
        order = np.sort(order)
        for t in range(len(order)):
            oi[cnt[r] + t] = idx[a + order[t]]
            od[cnt[r] + t] = seg[order[t]]
    return cnt, oi, od


@njit(cache=True)
def topk_rows(indptr, indices, data, K):
    n = len(indptr) - 1
    ptr = np.zeros(n + 1, np.int64)
    for r in range(n):
        ptr[r + 1] = ptr[r] + min(K, indptr[r + 1] - indptr[r])
    orow = np.empty(ptr[n], np.int32)
    ocol = np.empty(ptr[n], np.int32)
    osc = np.empty(ptr[n], np.float32)
    ork = np.empty(ptr[n], np.int16)
    for r in range(n):
        a, b = indptr[r], indptr[r + 1]
        m = b - a
        if m == 0:
            continue
        vals = data[a:b]
        if m > K:
            kth = np.partition(vals.copy(), m - K)[m - K]
            sel = np.empty(K, np.int64)
            c = 0
            for j in range(m):
                if vals[j] > kth:
                    sel[c] = j
                    c += 1
            for j in range(m):
                if c >= K:
                    break
                if vals[j] == kth:
                    sel[c] = j
                    c += 1
        else:
            sel = np.arange(m)
            c = m
        sub = np.empty(c, vals.dtype)
        for t in range(c):
            sub[t] = vals[sel[t]]
        order = np.argsort(-sub)
        base = ptr[r]
        for t in range(c):
            j = sel[order[t]]
            orow[base + t] = r
            ocol[base + t] = indices[a + j]
            osc[base + t] = vals[j]
            ork[base + t] = t + 1
    return orow, ocol, osc, ork


@njit(cache=True)
def update_rev(indptr, indices, data, row_offset, best_s, best_q):
    R = best_s.shape[1]
    for r in range(len(indptr) - 1):
        q = row_offset + r
        for j in range(indptr[r], indptr[r + 1]):
            d = indices[j]
            v = data[j]
            if v <= best_s[d, R - 1]:
                continue
            k = R - 1
            while k > 0 and best_s[d, k - 1] < v:
                best_s[d, k] = best_s[d, k - 1]
                best_q[d, k] = best_q[d, k - 1]
                k -= 1
            best_s[d, k] = v
            best_q[d, k] = q


_G = {}


def _init(Q, DT, k1, rev_k, chunk):
    _G.update(Q=Q, DT=DT, k1=k1, rev_k=rev_k, chunk=chunk)


def _run(bounds):
    lo, hi = bounds
    Q, DT = _G["Q"], _G["DT"]
    nd = DT.shape[1]
    best_s = np.zeros((nd, _G["rev_k"]), np.float32)
    best_q = np.full((nd, _G["rev_k"]), -1, np.int32)
    parts = []
    touched = 0
    for s in range(lo, hi, _G["chunk"]):
        e = min(hi, s + _G["chunk"])
        S = (Q[s:e] @ DT).tocsr()
        S.data = S.data.astype(np.float32, copy=False)
        touched += S.nnz
        r, c, sc, rk = topk_rows(S.indptr, S.indices, S.data, _G["k1"])
        parts.append((r + s, c, sc, rk))
        update_rev(S.indptr, S.indices, S.data, s, best_s, best_q)
    keep = best_q >= 0
    dd = np.nonzero(keep)
    rev = (best_q[keep], dd[0].astype(np.int32), best_s[keep])
    fwd = tuple(np.concatenate([p[i] for p in parts]) if parts else np.zeros(0) for i in range(4))
    return fwd, rev, touched


def _l2_rows(M):
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms) @ M


def _merge_rev(q, d, s, rev_k):
    order = np.lexsort((-s, d))
    q, d, s = q[order], d[order], s[order]
    first = np.r_[True, d[1:] != d[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(len(d)), 0))
    keep = (np.arange(len(d)) - grp_start) < rev_k
    return q[keep], d[keep], s[keep]


def retrieve_country(rec, q_rows, d_rows, cfg, n_jobs, log):
    args = (rec["nt_p"], rec["nt_d"], rec["n_concat_p"], rec["n_concat_b"], rec["at_p"],
            rec["at_d"], rec["nu_p"], rec["nu_d"], rec["n_core_p"], rec["n_core_b"])
    qp, qf = csr_sort_unique(*build_feats(q_rows, *args))
    dp, dfe = csr_sort_unique(*build_feats(d_rows, *args))
    uniq, inv = np.unique(dfe, return_inverse=True)
    dfreq = np.bincount(inv, minlength=len(uniq))
    ftype = (uniq >> 40).astype(np.int64)
    caps = np.array([0] + [cfg["cap_" + t] for t in TYPES[1:]], np.float64)
    tw = np.array([0.0] + [cfg["w_" + t] for t in TYPES[1:]])
    w = tw[ftype] * np.log((len(d_rows) + 1.0) / dfreq)
    w[dfreq > caps[ftype]] = 0.0
    nf = len(uniq)
    D = sp.csr_matrix((w[inv].astype(np.float32), inv.astype(np.int32), dp), shape=(len(d_rows), nf))
    D.eliminate_zeros()
    D = _l2_rows(D).tocsr().astype(np.float32)
    pos = np.clip(np.searchsorted(uniq, qf), 0, nf - 1)
    found = uniq[pos] == qf
    qw = np.where(found, w[pos], 0.0).astype(np.float32)
    Q = sp.csr_matrix((qw, pos.astype(np.int32), qp), shape=(len(q_rows), nf))
    Q.eliminate_zeros()
    Q = _l2_rows(Q).tocsr().astype(np.float32)
    qptr, qidx, qdat = _prefix_filter(Q.indptr.astype(np.int64), Q.indices, Q.data, cfg["prefix_m"])
    Q = sp.csr_matrix((qdat, qidx, qptr), shape=Q.shape)
    DT = D.T.tocsr()
    nq = len(q_rows)
    n_tasks = max(1, min(n_jobs * 4, (nq + cfg["chunk_rows"] - 1) // cfg["chunk_rows"]))
    edges = np.linspace(0, nq, n_tasks + 1).astype(np.int64)
    bounds = [(int(edges[i]), int(edges[i + 1])) for i in range(n_tasks) if edges[i + 1] > edges[i]]
    with Pool(n_jobs, initializer=_init, initargs=(Q, DT, cfg["k1"], cfg["rev_k"], cfg["chunk_rows"])) as pool:
        results = pool.map(_run, bounds, chunksize=1)
    fr = np.concatenate([r[0][0] for r in results]).astype(np.int64)
    fc = np.concatenate([r[0][1] for r in results]).astype(np.int64)
    fs = np.concatenate([r[0][2] for r in results]).astype(np.float32)
    fk = np.concatenate([r[0][3] for r in results]).astype(np.int16)
    rq, rd, rs = _merge_rev(np.concatenate([r[1][0] for r in results]).astype(np.int64),
                            np.concatenate([r[1][1] for r in results]).astype(np.int64),
                            np.concatenate([r[1][2] for r in results]).astype(np.float32),
                            cfg["rev_k"])
    log(f"    avg pool records touched per S1: {sum(r[2] for r in results) / max(nq, 1):.0f}")
    a = np.concatenate([q_rows[fr], q_rows[rq]])
    b = np.concatenate([d_rows[fc], d_rows[rd]])
    s = np.concatenate([fs, rs])
    k = np.concatenate([fk, np.full(len(rq), cfg["k1"] + 1, np.int16)])
    return a, b, s, k


def retrieve_split(split, work, cfg, n_jobs, log=print):
    t0 = time.time()
    rec = work.load_arrays(f"{split}/rec", ["nt_p", "nt_d", "n_concat_p", "n_concat_b", "at_p",
                                            "at_d", "nu_p", "nu_d", "n_core_p", "n_core_b", "country"])
    meta = work.load_json(f"{split}/meta.json")
    n1 = meta["n1"]
    N = len(rec["country"])
    country = rec["country"]
    parts = []
    for c, cname in enumerate(meta["countries"]):
        q_rows = np.nonzero(country[:n1] == c)[0].astype(np.int64)
        d_rows = (n1 + np.nonzero(country[n1:] == c)[0]).astype(np.int64)
        if len(q_rows) == 0 or len(d_rows) == 0:
            continue
        parts.append(retrieve_country(rec, q_rows, d_rows, cfg, n_jobs, log))
        log(f"[{split}] retrieved {cname}: {len(q_rows)} S1 x {len(d_rows)} pool "
            f"-> {len(parts[-1][0])} pairs ({time.time() - t0:.0f}s)")
    a = np.concatenate([p[0] for p in parts])
    b = np.concatenate([p[1] for p in parts])
    s = np.concatenate([p[2] for p in parts])
    k = np.concatenate([p[3] for p in parts])
    # dedupe (forward entries come first, so np.unique keeps their rank)
    _, first = np.unique(a * N + b, return_index=True)
    a, b, s, k = a[first].astype(np.int32), b[first].astype(np.int32), s[first], k[first]
    cand = {"a": a, "b": b, "score": s, "fwd_rank": k}
    cand.update(retrieval_context(a, b, s, n1, N))
    if split == "train":
        truth = work.load_arrays("train/truth")
        cand["y"] = (truth["pool_true"][b] == a).astype(np.int8)
        report_recall(cand["y"], truth["n_true"], a, country[:n1], meta["countries"], log, "retrieval")
    work.save_arrays(f"{split}/cand_raw", cand)
    log(f"[{split}] block done: {len(a)} pairs, {len(a) / n1:.1f} per S1 ({time.time() - t0:.0f}s)")


def retrieval_context(a, b, s, n1, N):
    rev_rank = group_rank_desc(b.astype(np.int64), s.astype(np.float64))
    b_top1, _, _, _, b_size = group_top2(b.astype(np.int64), s.astype(np.float64), N)
    a_top1, _, _, _, _ = group_top2(a.astype(np.int64), s.astype(np.float64), n1)
    return {"rev_rank": rev_rank.astype(np.int16), "n_cand_b": b_size[b].astype(np.int16),
            "gap_a": (a_top1[a] - s).astype(np.float32), "gap_b": (b_top1[b] - s).astype(np.float32)}


def report_recall(y, n_true, a, s1_country, countries, log, stage):
    total = n_true.sum()
    found = np.bincount(a[y == 1], minlength=len(n_true))
    msg = [f"{stage} pair recall={found.sum() / max(total, 1):.4f}"]
    for c, name in enumerate(countries):
        m = s1_country == c
        if n_true[m].sum():
            msg.append(f"{name}={found[m].sum() / n_true[m].sum():.4f}")
    log("  " + "  ".join(msg))
