"""Precision engine: calibration, one-S1-per-record exclusivity, the has-match model,
expected-F0.5 set selection, and the challenge metric (macro F0.5 per S1; empty-vs-empty = 1).

Pairs never cross countries (blocking runs per country), so every setting can be chosen per country
and a pair's country is its S1's country."""
import time

import numpy as np
from numba import njit, prange

HM_FEATURES = ["q1", "q2", "q3", "q_sum", "n_q05", "n_q01", "n_cand", "q_entropy",
               "max_jw_sorted", "max_atok_cos", "max_num_shared", "name_freq_a", "addr_empty_a",
               "ntok_a"]
HM_MIN_PHAS = 0.05   # has-match rescaling: floor on 1 - prod(1 - q), i.e. the ratio is capped


# ------------------------------------------------------------------ metric
def f05_from_counts(tp, npred, nt):
    """Per-S1 F0.5 = 1.25 TP / (|P| + 0.25 |T|); empty prediction on a singleton scores 1."""
    tp, npred, nt = (np.asarray(x, np.float64) for x in (tp, npred, nt))
    denom = npred + 0.25 * nt
    f = np.divide(1.25 * tp, denom, out=np.zeros(len(nt)), where=denom > 0)
    f[npred == 0] = (nt[npred == 0] == 0).astype(np.float64)
    return f


def macro_f05(a, sel, y, n_true, s1_mask=None):
    n1 = len(n_true)
    f = f05_from_counts(np.bincount(a[sel & (y == 1)], minlength=n1),
                        np.bincount(a[sel], minlength=n1), n_true)
    if s1_mask is not None:
        f = f[s1_mask]
    return float(f.mean()) if len(f) else float("nan")


# ------------------------------------------------------------------ isotonic calibration
def _pav(y, w):
    vals, wts, cnts = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            wt = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / wt)
            wts.append(wt)
            cnts.append(c1 + c2)
    out = []
    for v, c in zip(vals, cnts):
        out.extend([v] * c)
    return np.array(out)


def fit_isotonic(p, y, n_bins=4000):
    """Isotonic regression (pool-adjacent-violators) on quantile bins of p."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 1)
    cnt = np.bincount(idx, minlength=len(edges)).astype(np.float64)
    ok = cnt > 0
    mean_p = np.bincount(idx, weights=p, minlength=len(edges))[ok] / cnt[ok]
    mean_y = np.bincount(idx, weights=y, minlength=len(edges))[ok] / cnt[ok]
    fitted = _pav(mean_y, cnt[ok])
    return {"x": mean_p.tolist(), "y": fitted.tolist()}


def apply_isotonic(p, iso):
    return np.interp(p, np.array(iso["x"]), np.array(iso["y"])).astype(np.float32)


def fit_calibration(p, y, pair_c, countries, min_pairs, log=print):
    """Isotonic per country with >= min_pairs pairs, pooled otherwise (and for unseen countries)."""
    p64, y64 = p.astype(np.float64), y.astype(np.float64)
    cal = {"pooled": fit_isotonic(p64, y64), "by_country": {}}
    for c, name in enumerate(countries):
        m = pair_c == c
        if m.sum() >= min_pairs:
            cal["by_country"][name] = fit_isotonic(p64[m], y64[m])
    log(f"  calibration: per country {sorted(cal['by_country'])}, others pooled")
    return cal


def calibrate(p, pair_c, countries, cal):
    q = apply_isotonic(p, cal["pooled"])
    for c, name in enumerate(countries):
        iso = cal["by_country"].get(name)
        if iso is not None:
            m = pair_c == c
            q[m] = apply_isotonic(p[m], iso)
    return q


def lam_shift(q, lam):
    """q' = lam q / (lam q + 1 - q): multiplies the odds by lam (lam may be per row)."""
    q = q.astype(np.float64)
    return (lam * q / (lam * q + 1.0 - q)).astype(np.float32)


# ------------------------------------------------------------------ exclusivity
@njit(cache=True)
def exclusive(b, p, N, delta):
    """Hard: each S2/S3 record keeps only its best S1, and only if it beats the runner-up by delta."""
    best_i = np.full(N, -1, np.int64)
    best_p = np.full(N, -1.0)
    second = np.full(N, -1.0)
    for i in range(len(b)):
        g = b[i]
        v = p[i]
        if v > best_p[g]:
            second[g] = best_p[g]
            best_p[g] = v
            best_i[g] = i
        elif v > second[g]:
            second[g] = v
    out = np.zeros(len(b), np.float32)
    for i in range(len(b)):
        g = b[i]
        if best_i[g] == i and p[i] - max(second[g], 0.0) >= delta:
            out[i] = p[i]
    return out


@njit(cache=True)
def soft_exclusive(b, p, N):
    """Soft: q'_a = o_a / (1 + sum_j o_j), o = q / (1 - q) over the record's candidate S1s -- the
    chance the record belongs to a, given it belongs to at most one S1. One candidate keeps q."""
    tot = np.zeros(N)
    o = np.empty(len(b))
    for i in range(len(b)):
        v = min(max(p[i], 0.0), 1.0 - 1e-6)
        o[i] = v / (1.0 - v)
        tot[b[i]] += o[i]
    out = np.empty(len(b), np.float32)
    for i in range(len(b)):
        out[i] = o[i] / (1.0 + tot[b[i]])
    return out


def excl_one(b, q, N, opt):
    """opt: "soft" or "hard:<delta>"."""
    b64, q64 = b.astype(np.int64), q.astype(np.float64)
    if opt == "soft":
        return soft_exclusive(b64, q64, N)
    return exclusive(b64, q64, N, float(opt.split(":")[1]))


def excl_by_country(b, q, N, pair_c, countries, opts, default):
    """Apply each country's exclusivity option to its pairs (records never cross countries)."""
    chosen = [opts.get(name, default) for name in countries]
    if len(set(chosen)) == 1:
        return excl_one(b, q, N, chosen[0])
    out = np.zeros(len(q), np.float32)
    for c, opt in enumerate(chosen):
        m = pair_c == c
        if m.any():
            out[m] = excl_one(b[m], q[m], N, opt)
    return out


# ------------------------------------------------------------------ expected F0.5 selection
@njit(cache=True)
def _ef_best_k(ps, p_has):
    """Best k for predicting the top k. p_has < 0: candidates independent (v1/v2). Otherwise the
    has-match model's P(at least one true match) rescales: E[empty] = 1 - p_has and
    E[top k] = p_has / (1 - prod(1 - q)) * the usual value (ratio capped via HM_MIN_PHAS)."""
    n = len(ps)
    suf = np.zeros((n + 1, n + 1))
    suf[n, 0] = 1.0
    for k in range(n - 1, -1, -1):
        pk = ps[k]
        for c in range(0, n - k + 1):
            v = suf[k + 1, c] * (1.0 - pk)
            if c > 0:
                v += suf[k + 1, c - 1] * pk
            suf[k, c] = v
    scale = 1.0
    if p_has >= 0.0:
        best = 1.0 - p_has
        scale = p_has / max(1.0 - suf[0, 0], HM_MIN_PHAS)
    else:
        best = suf[0, 0]  # predicting nothing scores 1 only if there is no true match
    best_k = 0
    pre = np.zeros(n + 1)
    pre[0] = 1.0
    for k in range(1, n + 1):
        pk = ps[k - 1]
        for c in range(k, 0, -1):
            pre[c] = pre[c] * (1.0 - pk) + pre[c - 1] * pk
        pre[0] *= 1.0 - pk
        e = 0.0
        for A in range(1, k + 1):
            if pre[A] < 1e-12:
                continue
            for Bc in range(0, n - k + 1):
                s = suf[k, Bc]
                if s < 1e-12:
                    continue
                e += pre[A] * s * 1.25 * A / (k + 0.25 * (A + Bc))
        e *= scale
        if e > best:
            best = e
            best_k = k
    return best_k


@njit(parallel=True, cache=True)
def _ef_select(starts, order, q, a, min_p, p_has, sel):
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        m = 0
        for i in range(s, e):
            if q[order[i]] >= min_p:
                m += 1
        if m == 0:
            continue
        ps = np.empty(m)
        idx = np.empty(m, np.int64)
        k = 0
        for i in range(s, e):  # order is sorted by q descending within the group
            if q[order[i]] >= min_p:
                ps[k] = q[order[i]]
                idx[k] = order[i]
                k += 1
        ph = p_has[a[order[s]]] if len(p_has) else -1.0
        best_k = _ef_best_k(ps, ph)
        for t in range(best_k):
            sel[idx[t]] = True


def _groups(a, q):
    order = np.lexsort((-q, a))
    ga = a[order]
    starts = np.r_[0, np.nonzero(ga[1:] != ga[:-1])[0] + 1, len(ga)].astype(np.int64)
    return order.astype(np.int64), starts


def select(a, q, min_p=0.001, p_has=None):
    """Expected-F0.5 set per S1. p_has: per-S1 P(has a true match), or None (off)."""
    a64 = a.astype(np.int64)
    order, starts = _groups(a64, q)
    sel = np.zeros(len(a), np.bool_)
    ph = np.zeros(0) if p_has is None else np.asarray(p_has, np.float64)
    _ef_select(starts, order, q.astype(np.float64), a64, min_p, ph, sel)
    return sel


# ------------------------------------------------------------------ has-match model
@njit(parallel=True, cache=True)
def _hm_feats(starts, order, q, jw, acos, nsh, out):
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        o = out[g]
        tot = 0.0
        c5 = 0
        c1 = 0
        mj, ma, mn = -1.0, -1.0, -1.0
        for r in range(s, e):
            i = order[r]
            v = q[i]
            tot += v
            if v > 0.5:
                c5 += 1
            if v > 0.1:
                c1 += 1
            if jw[i] > mj:
                mj = jw[i]
            if acos[i] > ma:
                ma = acos[i]
            if nsh[i] > mn:
                mn = nsh[i]
        ent = 0.0
        if tot > 0:
            for r in range(s, e):
                pr = q[order[r]] / tot
                if pr > 0:
                    ent -= pr * np.log(pr)
        o[0] = q[order[s]]
        o[1] = q[order[s + 1]] if e - s > 1 else 0.0
        o[2] = q[order[s + 2]] if e - s > 2 else 0.0
        o[3] = tot
        o[4] = c5
        o[5] = c1
        o[6] = e - s
        o[7] = ent
        o[8] = mj
        o[9] = ma
        o[10] = mn


HM_PAIR_COLS = ["jw_sorted", "atok_cos", "num_shared", "name_freq_a", "ntok_a"]


def hm_features(a, q, cols, a_empty):
    """Per-S1 features from its calibrated q (before exclusivity) -> (S1 ids, matrix).
    cols: {name: per-pair array} for HM_PAIR_COLS (pair features, see pairfeats)."""
    a64 = a.astype(np.int64)
    order, starts = _groups(a64, q)
    out = np.zeros((len(starts) - 1, len(HM_FEATURES)), np.float32)

    def f64(n):
        return np.nan_to_num(cols[n].astype(np.float64), nan=-1.0)
    _hm_feats(starts, order, q.astype(np.float64), f64("jw_sorted"), f64("atok_cos"),
              f64("num_shared"), out)
    first = order[starts[:-1]]
    s1 = a64[first]
    out[:, 11] = cols["name_freq_a"][first]
    out[:, 12] = a_empty[s1]
    out[:, 13] = cols["ntok_a"][first]
    return s1, out


def _hm_params(n_jobs, seed):
    return {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
            "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
            "num_threads": n_jobs, "seed": seed, "verbose": -1}


HM_ROUNDS = 300


def take_cols(X, idx, step=2_000_000):
    """Columns idx of a (memory-mapped) row-major matrix in one pass over its rows."""
    out = np.empty((X.shape[0], len(idx)), np.float32)
    for i in range(0, X.shape[0], step):
        out[i:i + step] = X[i:i + step][:, idx]
    return out


def fit_hasmatch(s1, F, n_true, folds, n_jobs, seed, log=print):
    """3-fold OOF P(has a true match) per S1 (on the model's S1 folds) + a final model on all.
    Returns (p_has per S1 id, -1 where the S1 has no candidate; final booster)."""
    import lightgbm as lgb
    y = (n_true[s1] > 0).astype(np.float32)
    oof = np.zeros(len(s1), np.float32)
    params = _hm_params(n_jobs, seed)
    fs = folds[s1]
    for k in np.unique(fs):
        tr = fs != k
        bst = lgb.train(params, lgb.Dataset(F[tr], y[tr]), num_boost_round=HM_ROUNDS)
        oof[~tr] = bst.predict(F[~tr], num_threads=n_jobs)
    final = lgb.train(params, lgb.Dataset(F, y), num_boost_round=HM_ROUNDS)
    p_has = np.full(len(n_true), -1.0, np.float32)
    p_has[s1] = oof
    brier = float(np.mean((oof - y) ** 2))
    log(f"  has-match model: {len(s1)} S1 with candidates, base rate {y.mean():.4f}, "
        f"OOF brier {brier:.4f}")
    return p_has, final


def predict_hasmatch(bst, s1, F, n1, n_jobs):
    p_has = np.full(n1, -1.0, np.float32)
    p_has[s1] = bst.predict(F, num_threads=n_jobs)
    return p_has


# ------------------------------------------------------------------ tuning (on OOF scores)
def excl_options(cfg):
    return [f"hard:{float(d):g}" for d in cfg["delta_grid"]] + (["soft"] if cfg["soft_excl"] else [])


def tune(a, b, q, y, n_true, N, s1_c, countries, cfg, p_has=None, log=print):
    """Grid over exclusivity options x has-match on/off, scored on OOF. Selection is separable by
    country, so each grid point gives per-country scores too; US/India then get their own best
    exclusivity option (has-match setting fixed at the global best), unseen countries the global."""
    t0 = time.time()
    pair_c = s1_c[a]
    opts = excl_options(cfg)
    hms = [False, True] if (cfg["has_match"] and p_has is not None) else [False]
    grid = []
    for opt in opts:
        qe = excl_one(b, q, N, opt)
        for hm in hms:
            sel = select(a, qe, cfg["min_p"], p_has if hm else None)
            row = {"excl": opt, "has_match": hm, "f05": macro_f05(a, sel, y, n_true)}
            for c, name in enumerate(countries):
                row[f"f05_{name}"] = macro_f05(a, sel, y, n_true, s1_c == c)
            grid.append(row)
            log(f"  grid {opt:>9} has_match={int(hm)}: f05={row['f05']:.5f}  " +
                "  ".join(f"{n}={row[f'f05_{n}']:.5f}" for n in countries) +
                f"  ({time.time() - t0:.0f}s)")
    best = max(grid, key=lambda r: r["f05"])
    by_country = {}
    for c, name in enumerate(countries):
        rows = [r for r in grid if r["has_match"] == best["has_match"]]
        by_country[name] = max(rows, key=lambda r: r[f"f05_{name}"])["excl"]
    dec = {"excl": best["excl"], "excl_by_country": by_country, "has_match": best["has_match"]}
    qe = excl_by_country(b, q, N, pair_c, countries, by_country, best["excl"])
    sel = select(a, qe, cfg["min_p"], p_has if best["has_match"] else None)
    dec["f05"] = macro_f05(a, sel, y, n_true)
    dec["f05_global_setting"] = best["f05"]
    dec["grid"] = grid
    log(f"  decision: global {best['excl']} has_match={best['has_match']} "
        f"(f05 {best['f05']:.5f}); per country {by_country} -> f05 {dec['f05']:.5f} "
        f"({time.time() - t0:.0f}s)")
    return dec, sel, qe


def decide(a, b, q, N, dec, pair_c, countries, min_p, p_has=None, lam=None):
    """Calibrated q -> (selected mask, q after exclusivity). lam: per-pair odds multiplier (the
    has-match probability, per S1, is shifted by the caller)."""
    if lam is not None:
        q = lam_shift(q, lam)
    qe = excl_by_country(b, q, N, pair_c, countries, dec["excl_by_country"], dec["excl"])
    sel = select(a, qe, min_p, p_has if dec["has_match"] else None)
    return sel, qe


# ------------------------------------------------------------------ prior shift (Saerens et al. 2002)
def prior_em(q, pi_tr, tol=1e-5, max_iter=1000):
    """EM for the test match rate given calibrated q (trained at rate pi_tr) -> (pi, lambda)."""
    q = q.astype(np.float64)
    pi = pi_tr
    for _ in range(max_iter):
        r, s = pi / pi_tr, (1.0 - pi) / (1.0 - pi_tr)
        new = float(np.mean(r * q / (r * q + s * (1.0 - q))))
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    r, s = pi / pi_tr, (1.0 - pi) / (1.0 - pi_tr)
    return pi, r / s
