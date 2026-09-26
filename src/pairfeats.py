"""Pair features (numba, parallel over pairs). No country one-hot anywhere."""
import numpy as np
from numba import njit, prange

from .strsim import (jaro_winkler, lev_ratio, num_stats, plain_jaccard, prefix_ratio, set_stats,
                     tri_dice)

REC_FIELDS = ["n_core_p", "n_core_b", "n_sorted_p", "n_sorted_b", "n_concat_p", "n_concat_b",
              "n_alt_p", "n_alt_b", "a_sorted_p", "a_sorted_b", "nt_p", "nt_d", "pt_p", "pt_d",
              "at_p", "at_d", "nu_p", "nu_d", "nv_p", "nv_b", "n_idf", "a_idf", "legal", "state",
              "src", "is_domain", "is_indic", "masked", "a_empty", "n_ntok", "name_freq",
              "nn_p", "nn_d", "gw_p", "gw_d", "ini_p", "ini_b"]  # last 6: stored by v3 for v4

PAIR_FEATURES = [
    "jw_core", "lev_core", "jw_sorted", "lev_sorted", "jw_concat", "lev_concat", "prefix_concat",
    "tri_concat", "alt_best_jw", "ntok_jacc", "ntok_cos", "ntok_unm_a", "ntok_unm_b",
    "ntok_max_idf", "monge_elkan", "phon_jacc", "legal_cat", "ntok_a", "ntok_b", "name_freq_a",
    "name_freq_b", "is_domain_b", "is_indic_b", "src_b", "atok_jacc", "atok_cos", "atok_unm_a",
    "atok_unm_b", "atok_max_idf", "jw_addr", "lev_addr", "num_shared", "num_a", "num_b",
    "num_conflict", "num_fuzzy", "masked_b", "state_cat", "addr_empty_b", "tri_addr",
    # v4: name numbers, initials, group words
    "nn_shared", "nn_conflict", "acro", "gw_a_only", "gw_b_only",
]
CTX_FEATURES = ["ret_score", "fwd_rank", "rev_rank", "gap_a", "gap_b", "n_cand_b"]
BASE_FEATURES = PAIR_FEATURES + CTX_FEATURES   # what full_features computes
FS_FEATURES = ["fs_llr"]                        # filled by fsem after the per-country EM fit
V4_FEATURES = BASE_FEATURES + FS_FEATURES       # the stored X up to v5
# v5.5 pair features (v5.5 plan W1/W2): near-miss numbers, digits in order, domain coverage, soft
# token matches, sibling / churn words
V55_PAIR = ["num_near1", "num_lev1", "digits_eq", "digits_lev", "nn_near1", "dom_cover",
            "addr_soft_a", "addr_soft_b", "name_soft_idf", "sib_a_only", "sib_b_only",
            "churn_a_only", "churn_b_only"]
# v5.5 candidate-structure features (who else carries this name), computed per split in `features`
V55_CAND = ["nm_rec_cnt_a", "nm_rec_cnt_b", "a_same_name_cands", "a_same_name_noaddr",
            "b_name_claimants", "b_name_rank"]
V55_FEATURES = V55_PAIR + V55_CAND
FULL_FEATURES = V4_FEATURES + V55_FEATURES      # the columns of the stored X
V55_REC_FIELDS = ["nu_p", "nu_d", "nn_p", "nn_d", "adig_p", "adig_b", "is_domain", "n_core_p",
                  "n_core_b", "a_sorted_p", "a_sorted_b", "nt_p", "nt_d", "nv_p", "nv_b", "n_idf",
                  "sib_p", "sib_d", "ch_p", "ch_d"]
CHEAP_PAIR = ["jw_sorted", "ntok_cos", "atok_cos", "num_shared", "num_conflict", "state_cat"]
CHEAP_FEATURES = CHEAP_PAIR + CTX_FEATURES

# +1: higher value can only raise match probability; -1: can only lower it
MONOTONE = {
    "jw_core": 1, "lev_core": 1, "jw_sorted": 1, "lev_sorted": 1, "jw_concat": 1, "lev_concat": 1,
    "prefix_concat": 1, "tri_concat": 1, "alt_best_jw": 1, "ntok_jacc": 1, "ntok_cos": 1,
    "ntok_unm_a": -1, "ntok_unm_b": -1, "monge_elkan": 1, "phon_jacc": 1, "atok_jacc": 1,
    "atok_cos": 1, "atok_unm_a": -1, "atok_unm_b": -1, "jw_addr": 1, "lev_addr": 1, "tri_addr": 1,
    "num_shared": 1, "num_conflict": -1, "ret_score": 1, "fwd_rank": -1, "rev_rank": -1,
    "gap_a": -1, "gap_b": -1,
    "nn_shared": 1, "nn_conflict": -1, "acro": 1, "gw_a_only": -1, "gw_b_only": -1, "fs_llr": 1,
    "digits_eq": 1, "digits_lev": 1, "dom_cover": 1, "addr_soft_a": 1, "addr_soft_b": 1,
    "name_soft_idf": 1,
}


@njit(cache=True)
def _cat(x, y):
    if x < 0 and y < 0:
        return 0.0
    if x == y:
        return 1.0
    if x < 0 or y < 0:
        return 2.0
    return 3.0


@njit(cache=True)
def _monge_elkan(d, p, a, b, vp, vb):
    ia, ie = p[a], p[a + 1]
    jb, je = p[b], p[b + 1]
    if ia == ie or jb == je:
        return np.nan
    s1 = 0.0
    for i in range(ia, ie):
        t = d[i]
        best = 0.0
        for j in range(jb, je):
            u = d[j]
            v = 1.0 if t == u else jaro_winkler(vb, vp[t], vp[t + 1], vb, vp[u], vp[u + 1])
            if v > best:
                best = v
        s1 += best
    s2 = 0.0
    for j in range(jb, je):
        u = d[j]
        best = 0.0
        for i in range(ia, ie):
            t = d[i]
            v = 1.0 if t == u else jaro_winkler(vb, vp[u], vp[u + 1], vb, vp[t], vp[t + 1])
            if v > best:
                best = v
        s2 += best
    return 0.5 * (s1 / (ie - ia) + s2 / (je - jb))


@njit(cache=True)
def _cos(ss, sa, sb):
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    return ss / np.sqrt(sa * sb)


@njit(cache=True)
def _unm(ss, s):
    if s <= 0.0:
        return np.nan
    return (s - ss) / s


@njit(cache=True)
def _shared_sorted(d, p, a, b):
    """Sorted-unique sets of records a and b -> (n_shared, n_a, n_b)."""
    i, ie = p[a], p[a + 1]
    j, je = p[b], p[b + 1]
    na = ie - i
    nb = je - j
    sh = 0
    while i < ie and j < je:
        if d[i] == d[j]:
            sh += 1
            i += 1
            j += 1
        elif d[i] < d[j]:
            i += 1
        else:
            j += 1
    return sh, na, nb


@njit(cache=True)
def _is_acro(inip, inib, x, ncp, ncb, y):
    """x's initials (>= 2 letters) equal y's core name with spaces removed (AL vs Association
    de Lesperance)."""
    s, e = inip[x], inip[x + 1]
    if e - s < 2:
        return False
    k = s
    for q in range(ncp[y], ncp[y + 1]):
        c = ncb[q]
        if c == 32:
            continue
        if k >= e or inib[k] != c:
            return False
        k += 1
    return k == e


@njit(cache=True)
def name_pair(a, b, nnp, nnd, gwp, gwd, inip, inib, ncp, ncb):
    """v4 name features -> (nn_shared, nn_conflict, acro, gw_a_only, gw_b_only)."""
    sh, na, nb = _shared_sorted(nnd, nnp, a, b)
    conflict = 1.0 if (na > 0 and nb > 0 and sh == 0) else 0.0
    acro = 1.0 if (_is_acro(inip, inib, a, ncp, ncb, b) or
                   _is_acro(inip, inib, b, ncp, ncb, a)) else 0.0
    gsh, ga, gb = _shared_sorted(gwd, gwp, a, b)
    return float(sh), conflict, acro, float(ga - gsh), float(gb - gsh)


@njit(parallel=True, cache=True)
def _full(A, B, ncp, ncb, nsp, nsb, nxp, nxb, nap, nab, asp, asb, ntp, ntd, ptp, ptd, atp, atd,
          nup, nud, nvp, nvb, nidf, aidf, legal, state, src, isdom, isind, masked, aempty, nntok,
          nfreq, nnp, nnd, gwp, gwd, inip, inib, out):
    for i in prange(len(A)):
        a = A[i]
        b = B[i]
        o = out[i]
        o[0] = jaro_winkler(ncb, ncp[a], ncp[a + 1], ncb, ncp[b], ncp[b + 1])
        o[1] = lev_ratio(ncb, ncp[a], ncp[a + 1], ncb, ncp[b], ncp[b + 1])
        o[2] = jaro_winkler(nsb, nsp[a], nsp[a + 1], nsb, nsp[b], nsp[b + 1])
        o[3] = lev_ratio(nsb, nsp[a], nsp[a + 1], nsb, nsp[b], nsp[b + 1])
        o[4] = jaro_winkler(nxb, nxp[a], nxp[a + 1], nxb, nxp[b], nxp[b + 1])
        o[5] = lev_ratio(nxb, nxp[a], nxp[a + 1], nxb, nxp[b], nxp[b + 1])
        o[6] = prefix_ratio(nxb, nxp[a], nxp[a + 1], nxb, nxp[b], nxp[b + 1])
        o[7] = tri_dice(nxb, nxp[a], nxp[a + 1], nxb, nxp[b], nxp[b + 1])
        best = np.nan
        has_a = nap[a + 1] > nap[a]
        has_b = nap[b + 1] > nap[b]
        if has_b:
            best = jaro_winkler(ncb, ncp[a], ncp[a + 1], nab, nap[b], nap[b + 1])
        if has_a:
            v = jaro_winkler(nab, nap[a], nap[a + 1], ncb, ncp[b], ncp[b + 1])
            if np.isnan(best) or v > best:
                best = v
        o[8] = best
        sh, na, nb, ss, sa, sb, mx = set_stats(ntd, ntp, a, b, nidf)
        o[9] = sh / (na + nb - sh) if na + nb - sh > 0 else np.nan
        o[10] = _cos(ss, sa, sb)
        o[11] = _unm(ss, sa)
        o[12] = _unm(ss, sb)
        o[13] = mx
        o[14] = _monge_elkan(ntd, ntp, a, b, nvp, nvb)
        o[15] = plain_jaccard(ptd, ptp, a, b)
        o[16] = _cat(legal[a], legal[b])
        o[17] = nntok[a]
        o[18] = nntok[b]
        o[19] = np.log1p(nfreq[a])
        o[20] = np.log1p(nfreq[b])
        o[21] = isdom[b]
        o[22] = isind[b]
        o[23] = src[b]
        sh, na, nb, ss, sa, sb, mx = set_stats(atd, atp, a, b, aidf)
        o[24] = sh / (na + nb - sh) if na + nb - sh > 0 else np.nan
        o[25] = _cos(ss, sa, sb)
        o[26] = _unm(ss, sa)
        o[27] = _unm(ss, sb)
        o[28] = mx
        o[29] = jaro_winkler(asb, asp[a], asp[a + 1], asb, asp[b], asp[b + 1])
        o[30] = lev_ratio(asb, asp[a], asp[a + 1], asb, asp[b], asp[b + 1])
        ns, n_a, n_b, fz = num_stats(nud, nup, a, b)
        o[31] = ns
        o[32] = n_a
        o[33] = n_b
        o[34] = 1.0 if (n_a > 0 and n_b > 0 and ns == 0) else 0.0
        o[35] = fz
        o[36] = masked[b]
        o[37] = _cat(state[a], state[b])
        o[38] = aempty[b]
        # char trigrams of the sorted address: typos, glued/split tokens ("ste100" vs "suite 100")
        if asp[a + 1] > asp[a] and asp[b + 1] > asp[b]:
            o[39] = tri_dice(asb, asp[a], asp[a + 1], asb, asp[b], asp[b + 1])
        else:
            o[39] = np.nan
        o[40], o[41], o[42], o[43], o[44] = name_pair(a, b, nnp, nnd, gwp, gwd, inip, inib,
                                                      ncp, ncb)


@njit(parallel=True, cache=True)
def _cheap(A, B, nsp, nsb, ntp, ntd, atp, atd, nup, nud, nidf, aidf, state, out):
    for i in prange(len(A)):
        a = A[i]
        b = B[i]
        o = out[i]
        o[0] = jaro_winkler(nsb, nsp[a], nsp[a + 1], nsb, nsp[b], nsp[b + 1])
        sh, na, nb, ss, sa, sb, mx = set_stats(ntd, ntp, a, b, nidf)
        o[1] = _cos(ss, sa, sb)
        sh, na, nb, ss, sa, sb, mx = set_stats(atd, atp, a, b, aidf)
        o[2] = _cos(ss, sa, sb)
        ns, n_a, n_b, fz = num_stats(nud, nup, a, b)
        o[3] = ns
        o[4] = 1.0 if (n_a > 0 and n_b > 0 and ns == 0) else 0.0
        o[5] = _cat(state[a], state[b])


def _ctx(cand):
    return np.column_stack([cand["score"], cand["fwd_rank"], cand["rev_rank"], cand["gap_a"],
                            cand["gap_b"], cand["n_cand_b"]]).astype(np.float32)


def full_features(rec, cand, extra=0):
    """BASE_FEATURES, plus `extra` trailing columns left NaN for the caller (no second copy)."""
    A = cand["a"].astype(np.int64)
    B = cand["b"].astype(np.int64)
    npf = len(PAIR_FEATURES)
    out = np.empty((len(A), len(BASE_FEATURES) + extra), np.float32)
    _full(A, B, *[rec[k] for k in REC_FIELDS], out)   # writes columns [0, npf) of each row
    out[:, npf:len(BASE_FEATURES)] = _ctx(cand)
    out[:, len(BASE_FEATURES):] = np.nan
    return out


def pair_features(rec, a, b):
    """PAIR_FEATURES only (no retrieval context), for arbitrary pairs (diagnostics)."""
    A, B = np.asarray(a, np.int64), np.asarray(b, np.int64)
    out = np.empty((len(A), len(PAIR_FEATURES)), np.float32)
    if len(A):
        _full(A, B, *[rec[k] for k in REC_FIELDS], out)
    return out


def cheap_features(rec, cand):
    A = cand["a"].astype(np.int64)
    B = cand["b"].astype(np.int64)
    out = np.empty((len(A), len(CHEAP_PAIR)), np.float32)
    _cheap(A, B, rec["n_sorted_p"], rec["n_sorted_b"], rec["nt_p"], rec["nt_d"], rec["at_p"],
           rec["at_d"], rec["nu_p"], rec["nu_d"], rec["n_idf"], rec["a_idf"], rec["state"], out)
    return np.hstack([out, _ctx(cand)])


# ------------------------------------------------------------------ v5.5 pair features
@njit(cache=True)
def _digits(x, out):
    """Decimal digits of x >= 0 into out, most significant first -> count."""
    n = 0
    tmp = np.empty(20, np.int64)
    if x == 0:
        tmp[0] = 0
        n = 1
    while x > 0:
        tmp[n] = x % 10
        x //= 10
        n += 1
    for k in range(n):
        out[k] = tmp[n - 1 - k]
    return n


@njit(cache=True)
def _near1(x, y, dx, dy):
    """Same length (>= 2 digits), one digit different or two adjacent digits swapped."""
    nx = _digits(x, dx)
    ny = _digits(y, dy)
    if nx != ny or nx < 2:
        return False
    first = -1
    mism = 0
    for k in range(nx):
        if dx[k] != dy[k]:
            mism += 1
            if first < 0:
                first = k
    if mism == 1:
        return True
    if mism == 2 and first + 1 < nx and dx[first] == dy[first + 1] and dx[first + 1] == dy[first]:
        return True
    return False


@njit(cache=True)
def _del1(big, nb, small, ns):
    """small equals big with one digit deleted."""
    i = 0
    j = 0
    skipped = False
    while i < nb and j < ns:
        if big[i] == small[j]:
            i += 1
            j += 1
        elif not skipped:
            skipped = True
            i += 1
        else:
            return False
    return True


@njit(cache=True)
def _lev1_len(x, y, dx, dy):
    """Lengths differ by one (>= 2 digits in the shorter) and one deletion makes them equal."""
    nx = _digits(x, dx)
    ny = _digits(y, dy)
    if nx == ny + 1 and ny >= 2:
        return _del1(dx, nx, dy, ny)
    if ny == nx + 1 and nx >= 2:
        return _del1(dy, ny, dx, nx)
    return False


@njit(cache=True)
def _num_near(d, p, a, b, dx, dy):
    """-> (near1, lev1) over pairs of different numbers, NaN when either side has none."""
    if p[a + 1] == p[a] or p[b + 1] == p[b]:
        return np.nan, np.nan
    n1 = 0.0
    l1 = 0.0
    for i in range(p[a], p[a + 1]):
        x = d[i]
        for j in range(p[b], p[b + 1]):
            y = d[j]
            if x == y or x < 10 or y < 10:
                continue
            if n1 == 0.0 and _near1(x, y, dx, dy):
                n1 = 1.0
            if l1 == 0.0 and _lev1_len(x, y, dx, dy):
                l1 = 1.0
    return n1, l1


@njit(cache=True)
def _bytes_eq(b, s1, e1, s2, e2):
    if e1 - s1 != e2 - s2:
        return False
    for k in range(e1 - s1):
        if b[s1 + k] != b[s2 + k]:
            return False
    return True


@njit(cache=True)
def _tok_bounds(p, buf, r, ts, te):
    """Word boundaries of record r's space-separated byte string -> count (at most len(ts))."""
    nt = 0
    cs = p[r]
    for q in range(p[r], p[r + 1] + 1):
        if q == p[r + 1] or buf[q] == 32:
            if q > cs and nt < len(ts):
                ts[nt] = cs
                te[nt] = q
                nt += 1
            cs = q + 1
    return nt


@njit(cache=True)
def _dom_cover(ncp, ncb, dom, oth):
    """Share of the domain stem `dom` covered by a left-to-right split into the other record's core
    words (full weight) or their initials (half weight): nmnidhiprivate <- nm nidhi private."""
    s, e = ncp[dom], ncp[dom + 1]
    L = e - s
    if L == 0 or L > 80:
        return np.nan
    ts = np.empty(32, np.int64)
    te = np.empty(32, np.int64)
    nt = _tok_bounds(ncp, ncb, oth, ts, te)
    if nt == 0:
        return np.nan
    dp = np.full(L + 1, -1.0)
    dp[0] = 0.0
    for i in range(L):
        if dp[i] < 0:
            continue
        if dp[i] > dp[i + 1]:
            dp[i + 1] = dp[i]
        for t in range(nt):
            tl = te[t] - ts[t]
            if i + tl <= L:
                ok = True
                for k in range(tl):
                    if ncb[s + i + k] != ncb[ts[t] + k]:
                        ok = False
                        break
                if ok and dp[i] + tl > dp[i + tl]:
                    dp[i + tl] = dp[i] + tl
            if ncb[s + i] == ncb[ts[t]] and dp[i] + 0.5 > dp[i + 1]:
                dp[i + 1] = dp[i] + 0.5
    return dp[L] / L


@njit(cache=True)
def _soft_side(buf, ts1, te1, n1, ts2, te2, n2):
    """Share of side-1 characters in words matched on side 2 exactly or by Jaro-Winkler >= 0.9."""
    tot = 0.0
    got = 0.0
    for i in range(n1):
        li = te1[i] - ts1[i]
        tot += li
        best = 0.0
        for j in range(n2):
            if _bytes_eq(buf, ts1[i], te1[i], ts2[j], te2[j]):
                best = 1.0
                break
            v = jaro_winkler(buf, ts1[i], te1[i], buf, ts2[j], te2[j])
            if v > best:
                best = v
        if best >= 0.9:
            got += li * best
    return got / tot if tot > 0 else np.nan


@njit(cache=True)
def _soft_dir(d, s1, e1, s2, e2, vp, vb, idf):
    w_all = 0.0
    w_got = 0.0
    for i in range(s1, e1):
        t = d[i]
        w = idf[t]
        w_all += w
        best = 0.0
        for j in range(s2, e2):
            u = d[j]
            if t == u:
                best = 1.0
                break
            v = jaro_winkler(vb, vp[t], vp[t + 1], vb, vp[u], vp[u + 1])
            if v > best:
                best = v
        if best >= 0.88:
            w_got += w * best
    return w_got / w_all if w_all > 0 else 0.0


@njit(cache=True)
def _name_soft_idf(d, p, a, b, vp, vb, idf):
    """IDF-weighted soft token match of the names, both directions averaged (JW >= 0.88 counts)."""
    ia, ie = p[a], p[a + 1]
    jb, je = p[b], p[b + 1]
    if ia == ie or jb == je:
        return np.nan
    return 0.5 * (_soft_dir(d, ia, ie, jb, je, vp, vb, idf) + _soft_dir(d, jb, je, ia, ie, vp, vb, idf))


@njit(parallel=True, cache=True)
def _v55(A, B, nup, nud, nnp, nnd, adp, adb, isdom, ncp, ncb, asp, asb, ntp, ntd, nvp, nvb, nidf,
         sbp, sbd, chp, chd, out):
    for i in prange(len(A)):
        a = A[i]
        b = B[i]
        o = out[i]
        dx = np.empty(20, np.int64)
        dy = np.empty(20, np.int64)
        n1, l1 = _num_near(nud, nup, a, b, dx, dy)
        o[0] = n1
        o[1] = l1
        if adp[a + 1] > adp[a] and adp[b + 1] > adp[b]:
            o[2] = 1.0 if _bytes_eq(adb, adp[a], adp[a + 1], adp[b], adp[b + 1]) else 0.0
            o[3] = lev_ratio(adb, adp[a], adp[a + 1], adb, adp[b], adp[b + 1])
        else:
            o[2] = np.nan
            o[3] = np.nan
        nn1, nl1 = _num_near(nnd, nnp, a, b, dx, dy)
        o[4] = nn1
        if isdom[b] == 1 and isdom[a] == 0:
            o[5] = _dom_cover(ncp, ncb, b, a)
        elif isdom[a] == 1 and isdom[b] == 0:
            o[5] = _dom_cover(ncp, ncb, a, b)
        else:
            o[5] = np.nan
        ts1 = np.empty(40, np.int64)
        te1 = np.empty(40, np.int64)
        ts2 = np.empty(40, np.int64)
        te2 = np.empty(40, np.int64)
        k1 = _tok_bounds(asp, asb, a, ts1, te1)
        k2 = _tok_bounds(asp, asb, b, ts2, te2)
        if k1 > 0 and k2 > 0:
            o[6] = _soft_side(asb, ts1, te1, k1, ts2, te2, k2)
            o[7] = _soft_side(asb, ts2, te2, k2, ts1, te1, k1)
        else:
            o[6] = np.nan
            o[7] = np.nan
        o[8] = _name_soft_idf(ntd, ntp, a, b, nvp, nvb, nidf)
        sh, na, nb = _shared_sorted(sbd, sbp, a, b)
        o[9] = na - sh
        o[10] = nb - sh
        sh, na, nb = _shared_sorted(chd, chp, a, b)
        o[11] = na - sh
        o[12] = nb - sh


def v55_pair_features(rec, a, b, out=None):
    """V55_PAIR columns for pairs (a, b); writes into `out` (n x len(V55_PAIR)) when given."""
    A, B = np.asarray(a, np.int64), np.asarray(b, np.int64)
    if out is None:
        out = np.empty((len(A), len(V55_PAIR)), np.float32)
    if len(A):
        _v55(A, B, *[rec[k] for k in V55_REC_FIELDS], out)
    return out


@njit(cache=True)
def name_keys(p, buf, pool):
    """FNV-1a hash of each record's sorted core name, combined with its pool (0 = empty name)."""
    n = len(p) - 1
    out = np.zeros(n, np.int64)
    for r in range(n):
        if p[r + 1] == p[r]:
            continue
        h = np.uint64(1469598103934665603)
        for k in range(p[r], p[r + 1]):
            h ^= np.uint64(buf[k])
            h *= np.uint64(1099511628211)
        h ^= np.uint64(pool[r] + 1) * np.uint64(0x9E3779B97F4A7C15)
        out[r] = np.int64(h >> np.uint64(1)) | np.int64(1)
    return out


def _count_of(keys, uk, cnt):
    if len(uk) == 0:
        return np.zeros(len(keys), np.int64)
    pos = np.clip(np.searchsorted(uk, keys), 0, len(uk) - 1)
    return np.where((uk[pos] == keys) & (keys != 0), cnt[pos], 0)


def candidate_structure(a, b, keys, n1, jw_sorted, atok_cos, num_shared, addr_empty_b):
    """V55_CAND columns: how many pool records carry each side's name, how many of a's other
    candidates look like a by name (and how many of those have no address), how many other S1s
    claim b by name, and a's rank among b's S1s by name, then address."""
    from .nbutils import group_rank_desc
    a64, b64 = a.astype(np.int64), b.astype(np.int64)
    N = len(keys)
    pk = keys[n1:]
    uk, cnt = np.unique(pk[pk != 0], return_counts=True)
    out = np.empty((len(a64), len(V55_CAND)), np.float32)
    out[:, 0] = np.log1p(_count_of(keys[a64], uk, cnt))
    out[:, 1] = np.log1p(np.maximum(_count_of(keys[b64], uk, cnt) - 1, 0))
    jw = np.nan_to_num(np.asarray(jw_sorted, np.float64), nan=0.0)
    same = jw >= 0.95
    noaddr = same & (np.asarray(addr_empty_b) > 0)
    ca = np.bincount(a64, weights=same.astype(np.float64), minlength=n1)
    cn = np.bincount(a64, weights=noaddr.astype(np.float64), minlength=n1)
    cb = np.bincount(b64, weights=same.astype(np.float64), minlength=N)
    out[:, 2] = ca[a64] - same
    out[:, 3] = cn[a64] - noaddr
    out[:, 4] = cb[b64] - same
    score = jw + 0.5 * np.nan_to_num(np.asarray(atok_cos, np.float64), nan=0.0) + 0.25 * (
        np.asarray(num_shared) > 0)
    out[:, 5] = group_rank_desc(b64, score)
    return out


@njit(parallel=True, cache=True)
def _twin(starts, order, B, p, keys, src, out):
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        for r in range(s, e):
            i = order[r]
            kb = keys[B[i]]
            sb = src[B[i]]
            best = 0.0
            if kb != 0:
                for r2 in range(s, e):
                    j = order[r2]
                    if j != i and keys[B[j]] == kb and src[B[j]] == sb and p[j] > best:
                        best = p[j]
            out[i] = best


def twin_feature(a, b, p, keys, src):
    """g_twin: best score among a's other candidates with b's exact name and b's source."""
    a64, b64 = a.astype(np.int64), b.astype(np.int64)
    order = np.argsort(a64, kind="stable").astype(np.int64)
    ga = a64[order]
    starts = np.r_[0, np.nonzero(ga[1:] != ga[:-1])[0] + 1, len(ga)].astype(np.int64)
    out = np.zeros(len(a64), np.float32)
    if len(a64):
        _twin(starts, order, b64, np.asarray(p, np.float64), keys, np.asarray(src, np.int64), out)
    return out
