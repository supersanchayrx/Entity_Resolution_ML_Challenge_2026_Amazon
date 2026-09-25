"""Fellegi-Sunter with EM (book ch. 4), fitted per (split, country) -> stage-1 feature `fs_llr`.

Each pair's comparison vector gamma has 8 comparisons with a few levels each (level 0 = best),
built from existing pair features. The model is the classic two-class mixture
    P(gamma) = lam * prod_k m_k(gamma_k) + (1 - lam) * prod_k u_k(gamma_k)
fitted by EM on the table of level-pattern counts (at most 30,720 patterns), starting from the
supervised values of the labelled training pairs. fs_llr = sum_k log(m_k / u_k) under the pair's
own (split, country) fit, so France gets French m and u without labels.
"""
import numpy as np

# (name, levels) -- level labels are for the log only
COMPARISONS = [
    ("name_jw", [">=.97", ">=.90", ">=.80", "lower"]),
    ("name_tok", [">=.9", ">=.6", ">=.3", "lower"]),
    ("addr_tok", [">=.9", ">=.6", ">=.3", "lower", "missing"]),
    ("addr_tri", [">=.8", ">=.5", "lower", "missing"]),
    ("house_num", ["shared", "conflict", "missing"]),
    ("state", ["same", "one missing", "different", "both missing"]),
    ("legal", ["same", "one missing", "different", "both missing"]),
    ("initials", ["yes", "no"]),
]
N_LEVELS = np.array([len(c[1]) for c in COMPARISONS], np.int64)
STRIDES = np.r_[np.cumprod(N_LEVELS[::-1])[::-1][1:], 1].astype(np.int64)
N_PATTERNS = int(np.prod(N_LEVELS))
CLIP = 1e-5
LAM_RANGE = (0.02, 0.5)
# _cat codes (0 both missing, 1 same, 2 one missing, 3 different) -> level
_CAT_LEVEL = np.array([3, 0, 1, 2], np.int64)


def _band(x, cuts):
    """Level 0 for x >= cuts[0], 1 for x >= cuts[1], ..., len(cuts) otherwise."""
    lev = np.full(len(x), len(cuts), np.int64)
    for i in range(len(cuts) - 1, -1, -1):
        lev[x >= cuts[i]] = i
    return lev


def patterns(X, names, step=5_000_000):
    """Pair features -> level-pattern index per pair (int32), in row chunks."""
    col = {n: names.index(n) for n in ("jw_sorted", "ntok_cos", "atok_cos", "tri_addr",
                                       "num_shared", "num_conflict", "state_cat", "legal_cat",
                                       "acro")}
    out = np.empty(X.shape[0], np.int32)
    for s in range(0, X.shape[0], step):
        C = {n: X[s:s + step, j].astype(np.float64) for n, j in col.items()}
        miss = np.isnan(C["tri_addr"])            # either address has no words
        lev = [
            _band(C["jw_sorted"], [0.97, 0.90, 0.80]),
            _band(C["ntok_cos"], [0.9, 0.6, 0.3]),
            np.where(miss, 4, _band(np.nan_to_num(C["atok_cos"]), [0.9, 0.6, 0.3])),
            np.where(miss, 3, _band(np.nan_to_num(C["tri_addr"]), [0.8, 0.5])),
            np.where(C["num_shared"] > 0, 0, np.where(C["num_conflict"] > 0, 1, 2)),
            _CAT_LEVEL[C["state_cat"].astype(np.int64)],
            _CAT_LEVEL[C["legal_cat"].astype(np.int64)],
            np.where(C["acro"] > 0, 0, 1),
        ]
        pat = np.zeros(len(lev[0]), np.int64)
        for k, l in enumerate(lev):
            pat += l * STRIDES[k]
        out[s:s + step] = pat
    return out


def decode(pat):
    """Pattern indices -> (n, K) level matrix."""
    pat = np.asarray(pat, np.int64)
    return np.stack([(pat // STRIDES[k]) % N_LEVELS[k] for k in range(len(N_LEVELS))], axis=1)


def _norm(v):
    """Counts or weights -> probabilities clipped to [CLIP, 1 - CLIP]."""
    v = np.asarray(v, np.float64)
    v = v / max(v.sum(), 1e-300)
    v = np.clip(v, CLIP, 1 - CLIP)
    return v / v.sum()


def supervised(pat, y):
    """m, u and lam from labelled pairs."""
    cnt1 = np.bincount(pat[y == 1], minlength=N_PATTERNS).astype(np.float64)
    cnt0 = np.bincount(pat[y == 0], minlength=N_PATTERNS).astype(np.float64)
    G = decode(np.arange(N_PATTERNS))
    m = [_norm(np.bincount(G[:, k], cnt1, minlength=N_LEVELS[k])) for k in range(len(N_LEVELS))]
    u = [_norm(np.bincount(G[:, k], cnt0, minlength=N_LEVELS[k])) for k in range(len(N_LEVELS))]
    return {"m": m, "u": u, "lam": float(cnt1.sum() / max(cnt1.sum() + cnt0.sum(), 1.0))}


def em(counts, init, max_iter=200, tol=1e-7):
    """EM on the pattern-count table. tol applies to the mean log-likelihood per pair."""
    nz = np.nonzero(counts)[0]
    c = counts[nz].astype(np.float64)
    G = decode(nz)
    m = [np.array(v, np.float64) for v in init["m"]]
    u = [np.array(v, np.float64) for v in init["u"]]
    lam = float(np.clip(init["lam"], CLIP, 1 - CLIP))
    prev, it = -np.inf, 0
    for it in range(1, max_iter + 1):
        lm = np.log(lam) + sum(np.log(m[k][G[:, k]]) for k in range(len(m)))
        lu = np.log(1 - lam) + sum(np.log(u[k][G[:, k]]) for k in range(len(u)))
        mx = np.maximum(lm, lu)
        lp = mx + np.log(np.exp(lm - mx) + np.exp(lu - mx))
        g = np.exp(lm - lp)
        ll = float((c * lp).sum() / c.sum())
        wm, wu = c * g, c * (1 - g)
        lam = float(np.clip(wm.sum() / c.sum(), CLIP, 1 - CLIP))
        m = [_norm(np.bincount(G[:, k], wm, minlength=N_LEVELS[k])) for k in range(len(m))]
        u = [_norm(np.bincount(G[:, k], wu, minlength=N_LEVELS[k])) for k in range(len(u))]
        if abs(ll - prev) < tol:
            break
        prev = ll
    return {"m": m, "u": u, "lam": lam, "iters": it, "loglik": ll}


def guard(fit):
    """Reason to reject an EM fit, or None."""
    lo, hi = LAM_RANGE
    if not lo <= fit["lam"] <= hi:
        return f"lambda {fit['lam']:.4f} outside [{lo}, {hi}]"
    for k in (0, 1, 2):   # top name levels and top address level
        if fit["m"][k][0] < fit["u"][k][0]:
            return f"m < u at the top level of {COMPARISONS[k][0]}"
    return None


def llr_table(fit):
    """log(m/u) summed over comparisons, for every pattern."""
    G = decode(np.arange(N_PATTERNS))
    w = [np.log(np.asarray(fit["m"][k]) / np.asarray(fit["u"][k])) for k in range(len(N_LEVELS))]
    return sum(w[k][G[:, k]] for k in range(len(w))).astype(np.float32)


def _jsonable(fit):
    return {k: ([np.round(np.asarray(x), 6).tolist() for x in v] if k in ("m", "u") else v)
            for k, v in fit.items()}


def _log_fit(tag, fit, log):
    log(f"  fs/em {tag}: lambda={fit['lam']:.4f} iters={fit.get('iters', '-')} "
        f"source={fit.get('source', 'em')}")
    for k, (name, labels) in enumerate(COMPARISONS):
        cells = "  ".join(f"{lab}: m={fit['m'][k][i]:.3f} u={fit['u'][k][i]:.3f}"
                          for i, lab in enumerate(labels))
        log(f"      {name:<9} {cells}")


def fs_llr(X, names, pair_c, countries, y, sup, cfg, log=print, split=""):
    """fs_llr per pair for one split. y (training only) gives the supervised starting values,
    which are returned and reused for the test split (sup). -> (llr, sup, fits by country)."""
    pat = patterns(X, names)
    if sup is None:
        if y is None:
            raise ValueError("fs_llr: the supervised fit needs the labelled training split first")
        sup = supervised(pat, y)
        sup["source"] = "supervised"
        _log_fit("supervised (all training pairs)", sup, log)
    llr = np.zeros(len(pat), np.float32)
    fits = {}
    for c, name in enumerate(countries):
        mask = pair_c == c
        if not mask.any():
            continue
        counts = np.bincount(pat[mask], minlength=N_PATTERNS)
        fit = em(counts, sup, cfg["fs_max_iter"], cfg["fs_tol"])
        why = guard(fit)
        if why:
            log(f"  fs/em [{split}] {name}: FALLBACK to the supervised fit ({why})")
            fit = {**sup, "source": f"fallback: {why}", "em_lam": fit["lam"]}
        else:
            fit["source"] = "em"
        _log_fit(f"[{split}] {name} ({int(mask.sum())} pairs)", fit, log)
        llr[mask] = llr_table(fit)[pat[mask]]
        fits[name] = _jsonable(fit)
    return llr, sup, fits


def sup_to_json(sup):
    return _jsonable(sup)


def sup_from_json(d):
    return {"m": [np.array(v) for v in d["m"]], "u": [np.array(v) for v in d["u"]],
            "lam": d["lam"], "source": d.get("source", "supervised")}
