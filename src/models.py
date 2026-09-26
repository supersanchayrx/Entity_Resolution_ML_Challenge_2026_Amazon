"""Models: from-scratch logistic regression (re-ranker) and LightGBM with grouped CV."""
import time

import numpy as np

from .nbutils import group_rank_desc, group_top2


# ------------------------------------------------------------------ logistic regression
class LogReg:
    """L2-regularised logistic regression fitted by Newton's method (IRLS)."""

    def __init__(self, l2=1e-4, iters=30):
        self.l2, self.iters = l2, iters

    def fit(self, X, y):
        X = np.nan_to_num(X.astype(np.float64))
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        w = np.zeros(Z.shape[1])
        n = len(Z)
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
            g = Z.T @ (p - y) / n + self.l2 * w
            H = (Z * (p * (1 - p))[:, None]).T @ Z / n + self.l2 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-7:
                break
        self.w = w
        return self

    def decision(self, X):
        X = np.nan_to_num(X.astype(np.float64))
        return ((X - self.mu) / self.sd) @ self.w[:-1] + self.w[-1]

    def predict(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(self.decision(X), -30, 30)))

    def to_dict(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.mu, m.sd, m.w = (np.array(d[k]) for k in ("mu", "sd", "w"))
        return m


def rerank_design(X):
    """Cheap features -> LR design matrix (CHEAP_FEATURES order, see pairfeats)."""
    jw, ncos, acos, nsh, ncf, stc, sc, fr, rr, ga, gb, nb = (X[:, i] for i in range(12))
    return np.column_stack([jw, ncos, acos, np.minimum(nsh, 3), ncf, stc == 1, stc == 3, sc,
                            np.log1p(fr), np.log1p(rr), ga, gb, np.log1p(nb), jw * acos,
                            ncos * acos, sc * sc])


# ------------------------------------------------------------------ LightGBM
def _lgb_params(cfg, n_jobs, monotone, seed=None):
    seed = cfg["seed"] if seed is None else seed
    p = {
        "objective": "binary", "learning_rate": cfg["lgb_lr"], "num_leaves": cfg["lgb_leaves"],
        "min_data_in_leaf": cfg["lgb_min_leaf"], "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255, "num_threads": n_jobs,
        "seed": seed, "verbose": -1, "monotone_constraints": monotone,
        "monotone_constraints_method": "intermediate",
    }
    if cfg.get("lgb_deterministic"):
        # same data + params + thread count -> the same model (v5: the package must reproduce)
        p.update(deterministic=True, force_row_wise=True, bagging_seed=seed + 1,
                 feature_fraction_seed=seed + 2, data_random_seed=seed + 3)
    return p


def lgb_threads(cfg, n_jobs):
    return int(cfg.get("lgb_threads") or n_jobs)


def _deadline_stop(cfg, log):
    """v5 time guard for the fits themselves: once deadline_min - reserve_min has passed since the
    run's start (cfg["_t0"], set by the pipeline), stop the fit so tuning, prediction and the output
    files still happen before the session ends. Never fires in a run on schedule."""
    import lightgbm as lgb
    t0 = cfg.get("_t0")
    if not t0:
        return []
    limit = t0 + 60.0 * (cfg["deadline_min"] - cfg["reserve_min"])

    def cb(env):
        if env.iteration % 10 == 0 and time.time() > limit:
            log(f"  TIME GUARD: deadline reached, stopping this fit after {env.iteration + 1} rounds")
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list or [])
    cb.order = 40
    return [cb]


def fit_lgb(X, y, Xv, yv, cfg, n_jobs, monotone, rounds=None, log=print, seed=None, weight=None):
    import lightgbm as lgb
    params = _lgb_params(cfg, n_jobs, monotone, seed)
    dtr = lgb.Dataset(X, y, weight=weight, free_raw_data=True)
    guard = _deadline_stop(cfg, log)
    if Xv is not None:
        dv = lgb.Dataset(Xv, yv, reference=dtr)
        cbs = [lgb.early_stopping(cfg["lgb_early_stop"], verbose=False), lgb.log_evaluation(200)]
        bst = lgb.train(params, dtr, num_boost_round=rounds or cfg["lgb_rounds"], valid_sets=[dv],
                        callbacks=cbs + guard)
    else:
        bst = lgb.train(params, dtr, num_boost_round=rounds or cfg["lgb_rounds"], callbacks=guard)
    return bst


def take_rows(X, m):
    """X[m], or X itself when m selects every row (v5 fits all rows: saves a full copy)."""
    return X if m.all() else X[m]


def s1_folds(n1, k, seed):
    return np.random.RandomState(seed).randint(0, k, n1).astype(np.int8)


def cv_lgb(X, y, a, n1, cfg, n_jobs, monotone, log=print, tag="stage"):
    """Grouped-by-S1 K-fold -> (out-of-fold predictions, final model on a capped S1 sample).

    max_train_s1 = 0 uses every S1. cv_parallel fits the folds at once in threads (LightGBM drops the
    GIL), each with lgb_threads // n_folds threads; the final model then uses all lgb_threads. The
    random draws happen up front in the sequential order, so both modes see the same rows."""
    from concurrent.futures import ThreadPoolExecutor
    rng = np.random.RandomState(cfg["seed"])
    folds = s1_folds(n1, cfg["n_folds"], cfg["seed"])
    s1_with_pairs = np.unique(a)
    oof = np.zeros(len(y), np.float32)
    cap = int(cfg["max_train_s1"])
    threads = lgb_threads(cfg, n_jobs)

    def pick(pool):
        return pool if cap <= 0 or len(pool) <= cap else rng.choice(pool, cap, replace=False)

    def mask_of(s1s):
        m = np.zeros(n1, bool)
        m[s1s] = True
        return m[a]

    fold_s1 = [pick(s1_with_pairs[folds[s1_with_pairs] != k]) for k in range(cfg["n_folds"])]
    all_s1 = pick(s1_with_pairs)
    parallel = bool(cfg.get("cv_parallel")) and cfg["n_folds"] > 1
    fold_threads = max(1, threads // cfg["n_folds"]) if parallel else threads

    def run_fold(k):
        tr = fold_s1[k]
        n_es = max(1, len(tr) // 20)
        es, fit_s1 = tr[:n_es], tr[n_es:]
        m_fit, m_es = mask_of(fit_s1), mask_of(es)
        bst = fit_lgb(X[m_fit], y[m_fit], X[m_es], y[m_es], cfg, fold_threads, monotone, log=log)
        m_te = folds[a] == k
        oof[m_te] = bst.predict(X[m_te], num_iteration=bst.best_iteration, num_threads=fold_threads)
        log(f"  [{tag}] fold {k}: fit rows={m_fit.sum()} best_iter={bst.best_iteration}")
        return max(bst.best_iteration, 50)

    if parallel:
        log(f"  [{tag}] {cfg['n_folds']} folds in parallel, {fold_threads} threads each")
        with ThreadPoolExecutor(cfg["n_folds"]) as ex:
            iters = list(ex.map(run_fold, range(cfg["n_folds"])))
    else:
        iters = [run_fold(k) for k in range(cfg["n_folds"])]
    rounds = int(np.mean(iters) * 1.1)
    m_all = mask_of(all_s1)
    t0 = time.time()
    final = fit_lgb(take_rows(X, m_all), take_rows(y, m_all), None, None, cfg, threads, monotone,
                    rounds=rounds, log=log)
    final_min = (time.time() - t0) / 60
    log(f"  [{tag}] final model: rows={m_all.sum()} rounds={rounds} ({final_min:.1f} min)")
    return oof, final, {"rounds": rounds, "rows": all_s1, "final_min": final_min}


def extra_seed_models(X, y, a, n1, rows_s1, cfg, n_jobs, monotone, rounds, n_more, time_left, log,
                      est_min, tag="stage2"):
    """v5: n_more additional final models with other bagging/feature-sampling seeds (same rows and
    rounds as the first). time_left() -> minutes left for optional work; stops when the next fit
    would not fit, judged by est_min (the first model's fit time), then by the last fit's."""
    m = np.zeros(n1, bool)
    m[rows_s1] = True
    m = m[a]
    Xs, ys = take_rows(X, m), take_rows(y, m)
    models, dur = [], est_min
    for s in range(1, n_more + 1):
        if time_left() < dur:
            log(f"  [{tag}] time guard: {time_left():.0f} min left < {dur:.0f} min per seed; "
                f"stopping at {len(models) + 1} seeds")
            break
        t0 = time.time()
        models.append(fit_lgb(Xs, ys, None, None, cfg, lgb_threads(cfg, n_jobs), monotone,
                              rounds=rounds, log=log, seed=cfg["seed"] + 1000 * s))
        dur = (time.time() - t0) / 60
        log(f"  [{tag}] seed {s}: {rounds} rounds ({dur:.1f} min)")
    return models


# ------------------------------------------------------------------ stage-2 context
CTX2_FEATURES = ["p1", "p1_rank_a", "p1_other_a", "p1_sum_a", "p1_c05_a", "p1_rank_b",
                 "p1_other_b", "p1_margin_b", "p1_n_b"]
CTX2_MONOTONE = {"p1": 1, "p1_rank_a": -1, "p1_rank_b": -1, "p1_other_b": -1, "p1_margin_b": 1}


def p_context(a, b, p, n1, N):
    """Features describing how a pair's stage-1 score compares with its competitors."""
    a64, b64, p64 = a.astype(np.int64), b.astype(np.int64), p.astype(np.float64)
    rank_a = group_rank_desc(a64, p64)
    rank_b = group_rank_desc(b64, p64)
    t1a, t2a, suma, c05a, _ = group_top2(a64, p64, n1)
    t1b, t2b, _, _, szb = group_top2(b64, p64, N)
    other_a = np.maximum(np.where(p64 >= t1a[a64], t2a[a64], t1a[a64]), 0)
    other_b = np.maximum(np.where(p64 >= t1b[b64], t2b[b64], t1b[b64]), 0)
    return np.column_stack([p64, rank_a, other_a, suma[a64], c05a[a64], rank_b, other_b,
                            p64 - other_b, szb[b64]]).astype(np.float32)


def monotone_vector(names, table):
    return [int(table.get(n, 0)) for n in names]


# ------------------------------------------------------------------ v4 feature sets
def stage1_names(cfg):
    """Stage-1 columns of the stored X (FULL_FEATURES), minus fs_llr when feat_fs is off."""
    from .pairfeats import FS_FEATURES, FULL_FEATURES
    return [n for n in FULL_FEATURES if cfg["feat_fs"] or n not in FS_FEATURES]


def stage2_names(cfg, names1):
    from .groupfeats import GROUP_FEATURES, SOURCE_FEATURES
    return (names1 + CTX2_FEATURES + (GROUP_FEATURES if cfg["feat_group"] else [])
            + (SOURCE_FEATURES if cfg["feat_source"] else []))


def select_cols(X, all_names, names, step=2_000_000):
    """X restricted to `names` (X itself when nothing is dropped, else one row-chunked copy)."""
    if list(names) == list(all_names):
        return X
    idx = [all_names.index(n) for n in names]
    out = np.empty((X.shape[0], len(idx)), np.float32)
    for i in range(0, X.shape[0], step):
        out[i:i + step] = X[i:i + step][:, idx]
    return out


def stage2_matrix(X1, ctx, G, names2):
    """[stage-1 columns | p1 context | chosen group/source columns] in one allocation."""
    from .groupfeats import G_COLUMNS
    gcols = [G_COLUMNS.index(n) for n in names2 if n in G_COLUMNS]
    k1, kc = X1.shape[1], ctx.shape[1]
    out = np.empty((X1.shape[0], k1 + kc + len(gcols)), np.float32)
    step = 2_000_000
    for i in range(0, X1.shape[0], step):
        out[i:i + step, :k1] = X1[i:i + step]
    out[:, k1:k1 + kc] = ctx
    if gcols:
        out[:, k1 + kc:] = G[:, gcols]
    assert out.shape[1] == len(names2), (out.shape, len(names2))
    return out


def gain_ranks(bst, names, wanted):
    """1-based gain rank of each wanted feature present in the model."""
    order = np.argsort(-bst.feature_importance("gain"))
    rank = {names[j]: r + 1 for r, j in enumerate(order)}
    return {n: rank[n] for n in wanted if n in rank}
