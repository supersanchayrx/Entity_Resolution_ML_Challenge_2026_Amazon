"""v5.5 stacking (v5.5 plan W3, W7, W8): stages 1..n_stages, fold models on the test path, a holdout
scored exactly like test, per-country expert models and an unconstrained model blended with the
pooled last stage.

Every stage after the first is built by `next_stage` from the previous stage's scores: the same
function on train (OOF scores; holdout rows carry test-path scores) and on test (the mean of the
saved fold models), so train and test inputs are made the same way.

Model files (work/model): <stage>_f<fold>_s<seed>.txt (test_path=folds) or <stage>_final_s<seed>.txt
(test_path=final). The last stage adds <stage>x<country>_... (experts) and <stage>u_... (unconstrained).
model/stack.json lists the stages, their columns and files, and the blend weights.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .models import fit_lgb, lgb_threads, monotone_vector, p_context, s1_folds, take_rows

CTX_BASE = ["p", "p_rank_a", "p_other_a", "p_sum_a", "p_c05_a", "p_rank_b", "p_other_b",
            "p_margin_b", "p_n_b"]
CTX_MONO = {"p": 1, "p_rank_a": -1, "p_rank_b": -1, "p_other_b": -1, "p_margin_b": 1}


def holdout_s1(n1, cfg, s1_c=None, countries=None):
    """Per S1: True for the holdout (kept out of every fit, scored by the test path): a random
    holdout_frac of the S1s, plus every S1 of loco_countries (leave-one-country-out, given s1_c)."""
    f = float(cfg.get("holdout_frac") or 0.0)
    hold = (np.random.RandomState(int(cfg["seed"]) + 77).rand(n1) < f) if f > 0 else np.zeros(n1, bool)
    loco = cfg.get("loco_countries") or []
    if loco and s1_c is not None:
        codes = [countries.index(c) for c in loco if c in countries]
        hold |= np.isin(s1_c, codes)
    return hold


# ------------------------------------------------------------------ stage inputs
def ctx_names(level):
    """Context columns built from stage `level`'s score: p1, p1_rank_a, ... (v4 names for level 1)."""
    return [n.replace("p", f"p{level}", 1) for n in CTX_BASE]


def group_names(level, cfg):
    from .groupfeats import GROUP_FEATURES, SOURCE_FEATURES
    names = (GROUP_FEATURES if cfg["feat_group"] else []) + (SOURCE_FEATURES if cfg["feat_source"] else [])
    names = names if level == 1 else [f"{n}@{level}" for n in names]
    if cfg.get("feat_v55", True):
        names = names + [f"g_twin{level}"]
    return names


def mono_of(name, base_table):
    """Monotone direction of a column, by its base name (level suffixes stripped)."""
    from .groupfeats import GROUP_MONOTONE
    base = name.split("@")[0]
    if base.startswith("p") and base[1:2].isdigit():
        k = 2
        while base[k:k + 1].isdigit():
            k += 1
        return CTX_MONO.get("p" + base[k:], 0)
    return int({**base_table, **GROUP_MONOTONE}.get(base, 0))


def next_stage(Xprev, names_prev, p, level, a, b, n1, N, grec, keys, cfg):
    """[Xprev | context of p | group + source features of p | g_twin] and its column names."""
    from .groupfeats import G_COLUMNS, group_features
    from .pairfeats import twin_feature
    ctx = p_context(a, b, p, n1, N)
    G = group_features(a, b, p, grec, cfg)
    gnames = group_names(level, cfg)
    base = [n.split("@")[0] for n in gnames if not n.startswith("g_twin")]
    cols = [G[:, G_COLUMNS.index(n)] for n in base]
    if cfg.get("feat_v55", True):
        cols.append(twin_feature(a, b, p, keys, grec["src"]))
    k0 = Xprev.shape[1]
    out = np.empty((Xprev.shape[0], k0 + ctx.shape[1] + len(cols)), np.float32)
    step = 2_000_000
    for i in range(0, Xprev.shape[0], step):
        out[i:i + step, :k0] = Xprev[i:i + step]
    out[:, k0:k0 + ctx.shape[1]] = ctx
    for j, c in enumerate(cols):
        out[:, k0 + ctx.shape[1] + j] = c
    names = list(names_prev) + ctx_names(level) + gnames
    assert out.shape[1] == len(names), (out.shape, len(names))
    return out, names, G


# ------------------------------------------------------------------ CV with fold models
def cv_stage(X, y, a, n1, cfg, n_jobs, monotone, log, tag, hold, row_mask=None, seeds=1):
    """Grouped-by-S1 K-fold on the non-holdout S1s of row_mask (all rows if None).

    -> (pred, models, info). pred: OOF on non-holdout rows, the test path on holdout rows (the mean
    of the fold models, or the final model with test_path=final), NaN outside row_mask. models:
    [(file suffix, booster)] for the test path. Folds of the same S1 split as v5 (s1_folds)."""
    rng = np.random.RandomState(cfg["seed"])
    folds = s1_folds(n1, cfg["n_folds"], cfg["seed"])
    inm = np.ones(len(y), bool) if row_mask is None else row_mask
    s1s = np.unique(a[inm])
    s1s = s1s[~hold[s1s]]
    pred = np.full(len(y), np.nan, np.float32)
    cap = int(cfg["max_train_s1"])
    threads = lgb_threads(cfg, n_jobs)
    final_path = cfg.get("test_path", "folds") == "final"

    def pick(pool):
        return pool if cap <= 0 or len(pool) <= cap else rng.choice(pool, cap, replace=False)

    def mask_of(ids):
        m = np.zeros(n1, bool)
        m[ids] = True
        return m[a] & inm

    fold_s1 = [pick(s1s[folds[s1s] != k]) for k in range(cfg["n_folds"])]
    all_s1 = pick(s1s) if final_path else None
    parallel = bool(cfg.get("cv_parallel")) and cfg["n_folds"] > 1
    fth = max(1, threads // cfg["n_folds"]) if parallel else threads
    t0 = time.time()

    def run_fold(k):
        tr = fold_s1[k]
        n_es = max(1, len(tr) // 20)
        m_fit, m_es = mask_of(tr[n_es:]), mask_of(tr[:n_es])
        Xf, yf = X[m_fit], y[m_fit]
        bst = fit_lgb(Xf, yf, X[m_es], y[m_es], cfg, fth, monotone, log=log)
        best = max(bst.best_iteration, 50)
        ms = [(f"_f{k}_s0", bst, bst.best_iteration)]
        for s in range(1, seeds):
            if _minutes_left(cfg) < 20:
                log(f"  [{tag}] fold {k}: time guard, {len(ms)} seed(s)")
                break
            ms.append((f"_f{k}_s{s}", fit_lgb(Xf, yf, None, None, cfg, fth, monotone, rounds=best,
                                              log=log, seed=cfg["seed"] + 1000 * s), 0))
        del Xf, yf
        m_te = (folds[a] == k) & inm & ~hold[a]
        Xt = X[m_te]
        pred[m_te] = np.mean([m.predict(Xt, num_iteration=it or None, num_threads=fth)
                              for _, m, it in ms], axis=0)
        log(f"  [{tag}] fold {k}: fit rows={int(m_fit.sum())} best_iter={bst.best_iteration} "
            f"seeds={len(ms)} ({(time.time() - t0) / 60:.1f} min)")
        return best, ms

    if parallel:
        with ThreadPoolExecutor(cfg["n_folds"]) as ex:
            res = list(ex.map(run_fold, range(cfg["n_folds"])))
    else:
        res = [run_fold(k) for k in range(cfg["n_folds"])]
    iters = [r[0] for r in res]
    info = {"best_iters": iters, "rounds": int(np.mean(iters) * 1.1)}
    m_h = hold[a] & inm
    if final_path:
        m_all = mask_of(all_s1)
        Xa, ya = take_rows(X, m_all), take_rows(y, m_all)
        models = []
        for s in range(seeds):
            if s and _minutes_left(cfg) < 20:
                break
            models.append((f"_final_s{s}", fit_lgb(Xa, ya, None, None, cfg, threads, monotone,
                                                   rounds=info["rounds"], log=log,
                                                   seed=cfg["seed"] + 1000 * s), 0))
        del Xa, ya
        log(f"  [{tag}] final model(s): {len(models)} x {info['rounds']} rounds")
    else:
        models = [m for r in res for m in r[1]]
    if m_h.any():
        pred[m_h] = predict_models([(m, it) for _, m, it in models], X[m_h], threads)
    return pred, [(sfx, m) for sfx, m, _ in models], info


def predict_models(models, X, n_jobs):
    """Mean prediction of [(booster, num_iteration or 0)]."""
    return np.mean([m.predict(X, num_iteration=it or None, num_threads=n_jobs) for m, it in models],
                   axis=0).astype(np.float32)


def _minutes_left(cfg):
    t0 = cfg.get("_t0")
    if not t0:
        return 1e9
    return cfg["deadline_min"] - (time.time() - t0) / 60 - cfg["reserve_min"]


# ------------------------------------------------------------------ blend (W8)
def blend_weights(parts, y, mask, step=0.1):
    """Simplex weights (step) over the columns of parts (n x k) minimising log loss on mask."""
    P = np.clip(parts[mask].astype(np.float64), 1e-6, 1 - 1e-6)
    yy = y[mask].astype(np.float64)
    k = P.shape[1]
    n = int(round(1 / step))
    grid = [np.array(w, np.float64) / n for w in _simplex(k, n)]
    best, best_w = np.inf, None
    for w in grid:
        q = P @ w
        ll = -np.mean(yy * np.log(q) + (1 - yy) * np.log(1 - q))
        if ll < best:
            best, best_w = ll, w
    return best_w.tolist(), float(best)


def _simplex(k, n):
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in _simplex(k - 1, n - i):
            yield (i,) + rest


def apply_blend(p_pool, extras, pair_c, countries, blend):
    """Blend per training country; other countries (France) keep the pooled score.
    extras: {country name: [array per extra model] (NaN outside that country's rows)}."""
    out = p_pool.astype(np.float32).copy()
    for c, name in enumerate(countries):
        w = blend.get(name)
        if not w:
            continue
        m = pair_c == c
        cols = [p_pool[m]] + [e[m] for e in extras.get(name, [])]
        if len(cols) != len(w):
            raise ValueError(f"blend {name}: {len(w)} weights for {len(cols)} models")
        out[m] = np.sum([wi * ci for wi, ci in zip(w, cols)], axis=0)
    return out


def mono_vector_for(names, cfg, unconstrained=False):
    from .pairfeats import MONOTONE
    if unconstrained:
        return [0] * len(names)
    return [mono_of(n, MONOTONE) for n in names]


__all__ = ["holdout_s1", "next_stage", "cv_stage", "predict_models", "blend_weights", "apply_blend",
           "mono_vector_for", "ctx_names", "group_names", "monotone_vector"]
