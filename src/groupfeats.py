"""Stage-2 features from each S1's other likely matches (book ch. 7: one entity's records form a
cluster that should agree) and source-aware competitor scores (v4 plan section 4).

Both use the stage-1 score p1 only: OOF p1 on training, the final stage-1 model's p1 on test,
like the existing stage-2 context features. No label enters.

For pair (a, b), the likely set L(a) is a's candidates other than b with p1 >= g_min_p, the top
g_top by p1. Each b' in L(a) is compared with b on name words, address words, address trigrams and
numbers; the features summarise those comparisons, weighted by p1(b').
"""
import numpy as np
from numba import njit, prange

from .strsim import num_stats, set_stats, tri_dice

GROUP_FEATURES = ["g_n", "g_name_max", "g_name_wmean", "g_addr_max", "g_addr_wmean", "g_tri_wmean",
                  "g_num_agree", "g_num_conflict", "g_cons_num"]
SOURCE_FEATURES = ["s_best_same", "s_best_other", "s_n_same", "s_n_other"]
G_COLUMNS = GROUP_FEATURES + SOURCE_FEATURES
GROUP_MONOTONE = {"g_num_agree": 1, "g_num_conflict": -1, "g_cons_num": 1}
MAX_VOTE_NUMS = 8     # numbers per record entering the number vote


@njit(cache=True)
def _cos(ss, sa, sb):
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    return ss / np.sqrt(sa * sb)


@njit(cache=True)
def _has_num(nud, nup, r, x):
    for j in range(nup[r], nup[r + 1]):
        if nud[j] == x:
            return True
    return False


@njit(cache=True)
def _vote_add(vals, wts, nv, nud, nup, r, w):
    for j in range(nup[r], min(nup[r + 1], nup[r] + MAX_VOTE_NUMS)):
        x = nud[j]
        found = False
        for t in range(nv):
            if vals[t] == x:
                wts[t] += w
                found = True
                break
        if not found:
            vals[nv] = x
            wts[nv] = w
            nv += 1
    return nv


@njit(parallel=True, cache=True)
def _group(starts, order, A, B, p, src, ntp, ntd, nidf, atp, atd, aidf, asp, asb, nup, nud,
           min_p, top, out):
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        # likely prefix: rows sorted by p1 descending, so those with p1 >= min_p come first
        nl = 0
        for r in range(s, e):
            if p[order[r]] >= min_p:
                nl += 1
            else:
                break
        # source-aware: best two and count >= 0.5 per source (2, 3)
        b1 = np.empty(4)
        b1i = np.empty(4, np.int64)
        b2 = np.empty(4)
        c05 = np.empty(4, np.int64)
        for k in range(4):
            b1[k] = -1.0
            b1i[k] = -1
            b2[k] = -1.0
            c05[k] = 0
        for r in range(s, e):
            i = order[r]
            k = src[B[i]]
            v = p[i]
            if v >= 0.5:
                c05[k] += 1
            if v > b1[k]:
                b2[k] = b1[k]
                b1[k] = v
                b1i[k] = i
            elif v > b2[k]:
                b2[k] = v
        a = A[order[s]]
        vals = np.empty(MAX_VOTE_NUMS * (top + 1), np.int64)
        wts = np.empty(MAX_VOTE_NUMS * (top + 1))
        for r in range(s, e):
            i = order[r]
            bi = B[i]
            o = out[i]
            # ---- source-aware
            ks = src[bi]
            ko = 5 - ks          # 2 <-> 3
            best_same = b2[ks] if b1i[ks] == i else b1[ks]
            o[9] = max(best_same, 0.0)
            o[10] = max(b1[ko], 0.0)
            o[11] = c05[ks] - (1 if p[i] >= 0.5 else 0)
            o[12] = c05[ko]
            # ---- agreement with the likely set
            n = 0
            wsum = 0.0
            nmax = -1.0
            nw = 0.0
            amax = -1.0
            aw = 0.0
            tw = 0.0
            twsum = 0.0
            agree = 0.0
            confl = 0.0
            nv = _vote_add(vals, wts, 0, nud, nup, a, 1.0)
            b_has_tri = asp[bi + 1] > asp[bi]
            for r2 in range(s, s + nl):
                if n >= top:
                    break
                j = order[r2]
                if j == i:
                    continue
                bj = B[j]
                w = p[j]
                n += 1
                wsum += w
                _, _, _, ss, sa, sb, _ = set_stats(ntd, ntp, bi, bj, nidf)
                cn = _cos(ss, sa, sb)
                nmax = max(nmax, cn)
                nw += w * cn
                _, _, _, ss, sa, sb, _ = set_stats(atd, atp, bi, bj, aidf)
                ca = _cos(ss, sa, sb)
                amax = max(amax, ca)
                aw += w * ca
                if b_has_tri and asp[bj + 1] > asp[bj]:
                    tw += w * tri_dice(asb, asp[bi], asp[bi + 1], asb, asp[bj], asp[bj + 1])
                    twsum += w
                sh, na, nb, _ = num_stats(nud, nup, bi, bj)
                if sh > 0:
                    agree += w
                elif na > 0 and nb > 0:
                    confl += w
                nv = _vote_add(vals, wts, nv, nud, nup, bj, w)
            o[0] = n
            if n == 0:
                for t in range(1, 9):
                    o[t] = np.nan
                continue
            o[1] = nmax
            o[2] = nw / wsum
            o[3] = amax
            o[4] = aw / wsum
            o[5] = tw / twsum if twsum > 0 else np.nan
            o[6] = agree / wsum
            o[7] = confl / wsum
            if nv == 0 or nup[bi + 1] == nup[bi]:
                o[8] = np.nan
            else:
                bt = 0
                for t in range(1, nv):
                    if wts[t] > wts[bt]:
                        bt = t
                o[8] = 1.0 if _has_num(nud, nup, bi, vals[bt]) else 0.0


GROUP_REC_FIELDS = ["src", "nt_p", "nt_d", "n_idf", "at_p", "at_d", "a_idf", "a_sorted_p",
                    "a_sorted_b", "nu_p", "nu_d"]


def group_features(a, b, p1, rec, cfg):
    """-> (n_pairs, len(G_COLUMNS)) float32, columns G_COLUMNS."""
    a64, b64 = a.astype(np.int64), b.astype(np.int64)
    p = p1.astype(np.float64)
    order = np.lexsort((-p, a64)).astype(np.int64)
    ga = a64[order]
    starts = np.r_[0, np.nonzero(ga[1:] != ga[:-1])[0] + 1, len(ga)].astype(np.int64)
    out = np.empty((len(a64), len(G_COLUMNS)), np.float32)
    if len(a64):
        _group(starts, order, a64, b64, p, rec["src"].astype(np.int64), rec["nt_p"], rec["nt_d"],
               rec["n_idf"], rec["at_p"], rec["at_d"], rec["a_idf"], rec["a_sorted_p"],
               rec["a_sorted_b"], rec["nu_p"], rec["nu_d"], float(cfg["g_min_p"]),
               int(cfg["g_top"]), out)
    return out
