"""Pair features (numba, parallel over pairs). No country one-hot anywhere."""
import numpy as np
from numba import njit, prange

from .strsim import (jaro_winkler, lev_ratio, num_stats, plain_jaccard, prefix_ratio, set_stats,
                     tri_dice)

REC_FIELDS = ["n_core_p", "n_core_b", "n_sorted_p", "n_sorted_b", "n_concat_p", "n_concat_b",
              "n_alt_p", "n_alt_b", "a_sorted_p", "a_sorted_b", "nt_p", "nt_d", "pt_p", "pt_d",
              "at_p", "at_d", "nu_p", "nu_d", "nv_p", "nv_b", "n_idf", "a_idf", "legal", "state",
              "src", "is_domain", "is_indic", "masked", "a_empty", "n_ntok", "name_freq"]

PAIR_FEATURES = [
    "jw_core", "lev_core", "jw_sorted", "lev_sorted", "jw_concat", "lev_concat", "prefix_concat",
    "tri_concat", "alt_best_jw", "ntok_jacc", "ntok_cos", "ntok_unm_a", "ntok_unm_b",
    "ntok_max_idf", "monge_elkan", "phon_jacc", "legal_cat", "ntok_a", "ntok_b", "name_freq_a",
    "name_freq_b", "is_domain_b", "is_indic_b", "src_b", "atok_jacc", "atok_cos", "atok_unm_a",
    "atok_unm_b", "atok_max_idf", "jw_addr", "lev_addr", "num_shared", "num_a", "num_b",
    "num_conflict", "num_fuzzy", "masked_b", "state_cat", "addr_empty_b", "tri_addr",
]
CTX_FEATURES = ["ret_score", "fwd_rank", "rev_rank", "gap_a", "gap_b", "n_cand_b"]
FULL_FEATURES = PAIR_FEATURES + CTX_FEATURES
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


@njit(parallel=True, cache=True)
def _full(A, B, ncp, ncb, nsp, nsb, nxp, nxb, nap, nab, asp, asb, ntp, ntd, ptp, ptd, atp, atd,
          nup, nud, nvp, nvb, nidf, aidf, legal, state, src, isdom, isind, masked, aempty, nntok,
          nfreq, out):
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


def full_features(rec, cand):
    A = cand["a"].astype(np.int64)
    B = cand["b"].astype(np.int64)
    out = np.empty((len(A), len(PAIR_FEATURES)), np.float32)
    _full(A, B, *[rec[k] for k in REC_FIELDS], out)
    return np.hstack([out, _ctx(cand)])


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
