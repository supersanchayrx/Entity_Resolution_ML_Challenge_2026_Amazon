"""CLI entry point: python run.py <step> --data DATA --work WORK --out OUT [--set key=value ...]

Steps: make-sample | prepare | block | rerank | features | train | predict | variants | selftrain
       | diagnose | all (prepare .. diagnose; --from / --to limit the range) | tune (re-tune the
       decision step from saved OOF scores, minutes) | package (assemble the submission zip)

--profile NAME applies a named override set from config.PROFILES (v5, v5_fewcores, v5_midmem,
v5_lite, v4) before --set. --checkpoint DIR copies models, reports, scores and logs there after every
step (Kaggle: /kaggle/working survives the session).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from .config import load_config
from .io_utils import Work, read_ground_truth, read_source, write_id_lists

STEPS = ["prepare", "block", "rerank", "features", "train", "predict", "variants", "selftrain",
         "diagnose"]
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the run's start for the time guard: the notebook exports ER_T0 in its first cell
T0 = float(os.environ.get("ER_T0") or time.time())


def minutes_left(cfg):
    """Minutes left for optional work: deadline minus elapsed minus the reserve for the outputs."""
    return cfg["deadline_min"] - (time.time() - T0) / 60 - cfg["reserve_min"]


def guard(cfg, need_min, what, log):
    """True if an optional step of about need_min minutes still fits before the deadline."""
    left = minutes_left(cfg)
    if left < need_min:
        log(f"  TIME GUARD: skipping {what} ({left:.0f} min left after the reserve, needs ~{need_min})")
        return False
    return True


def _logger(work, step):
    path = work.w("logs", f"{time.strftime('%Y%m%d-%H%M%S')}-{step}.log")
    t0 = time.time()

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:7.0f}s] {msg}"
        print(line, flush=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


# ------------------------------------------------------------------ steps
def step_prepare(args, work, cfg, n_jobs, log):
    from .encode import prepare_split
    for split in _splits(args):
        prepare_split(split, args.data, work, cfg, n_jobs, log)


def step_block(args, work, cfg, n_jobs, log):
    from .retrieval import retrieve_split
    for split in _splits(args):
        retrieve_split(split, work, cfg, n_jobs, log)


def _splits(args):
    """--splits (v5.5 W9c): recompute only some splits, e.g. the test side with trained models."""
    return [s.strip() for s in (getattr(args, "splits", None) or "train,test").split(",") if s.strip()]


RERANK_LGB = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "verbose": -1}


def _rerank_extra(rec, a, b):
    """v5.5 re-ranker inputs beyond the LR design: name frequency, empty address, name lengths, domain."""
    return np.column_stack([np.log1p(rec["name_freq"][a]), rec["a_empty"][b], rec["n_ntok"][a],
                            rec["n_ntok"][b], rec["is_domain"][b]]).astype(np.float64)


def step_rerank(args, work, cfg, n_jobs, log):
    """Cut each S1's candidates to the top k2 (+ each record's best S1) by a re-ranker fitted on
    rerank_train_s1 training S1s: the from-scratch logistic regression (v1-v5), or with
    rerank_model=lgb a small LightGBM on the same design plus 5 columns (v5.5 W6)."""
    import lightgbm as lgb
    from .models import LogReg, rerank_design
    from .nbutils import group_rank_desc
    from .pairfeats import cheap_features
    from .retrieval import pool_of, report_recall
    rec_names = ["n_sorted_p", "n_sorted_b", "nt_p", "nt_d", "at_p", "at_d", "nu_p", "nu_d",
                 "n_idf", "a_idf", "state", "country", "name_freq", "a_empty", "n_ntok", "is_domain"]
    use_lgb = cfg.get("rerank_model", "lr") == "lgb"
    lr, gbm = None, None
    step = 20_000_000
    for split in _splits(args):
        rec = work.load_arrays(f"{split}/rec", rec_names)
        cand = work.load_arrays(f"{split}/cand_raw")
        X = cheap_features(rec, cand)
        a, b = cand["a"].astype(np.int64), cand["b"].astype(np.int64)
        meta = work.load_json(f"{split}/meta.json")

        def design(i, j):
            D = rerank_design(X[i:j])
            return np.hstack([D, _rerank_extra(rec, a[i:j], b[i:j])]) if use_lgb else D
        if split == "train":
            rng = np.random.RandomState(cfg["seed"])
            s1s = np.unique(a)
            s1s = s1s[~_hold(work, cfg)[s1s]]
            if len(s1s) > cfg["rerank_train_s1"]:
                s1s = rng.choice(s1s, cfg["rerank_train_s1"], replace=False)
            m = np.zeros(a.max() + 1, bool)
            m[s1s] = True
            m = m[a]
            idx = np.nonzero(m)[0]
            D = rerank_design(X[idx])
            lr = LogReg().fit(D, cand["y"][idx].astype(np.float64))
            work.save_json("model/rerank.json", lr.to_dict())
            if use_lgb:
                D = np.hstack([D, _rerank_extra(rec, a[idx], b[idx])])
                params = {**RERANK_LGB, "num_threads": n_jobs, "seed": cfg["seed"]}
                if cfg.get("lgb_deterministic"):
                    params.update(deterministic=True, force_row_wise=True)
                gbm = lgb.train(params, lgb.Dataset(D, cand["y"][idx].astype(np.float32)),
                                num_boost_round=int(cfg["rerank_rounds"]))
                gbm.save_model(work.w("model", "rerank_lgb.txt"))
            del D
        else:
            if lr is None:
                lr = LogReg.from_dict(work.load_json("model/rerank.json"))
            if use_lgb and gbm is None:
                gbm = lgb.Booster(model_file=work.r("model", "rerank_lgb.txt"))
        p_lr = np.concatenate([lr.predict(rerank_design(X[i:i + step])) for i in range(0, len(X), step)])
        if use_lgb:
            p = np.concatenate([gbm.predict(design(i, i + step), num_threads=n_jobs)
                                for i in range(0, len(X), step)])
        else:
            p = p_lr
        X = None
        keep = (group_rank_desc(a, p) <= cfg["k2"]) | (group_rank_desc(b, p) == 1)
        sub = {k: v[keep] for k, v in cand.items()}
        sub["rr_p"] = p[keep].astype(np.float32)
        work.save_arrays(f"{split}/cand", sub)
        log(f"[{split}] rerank ({'lgb' if use_lgb else 'lr'}) kept {keep.sum()} of {len(keep)} pairs "
            f"({keep.sum() / meta['n1']:.1f} per S1)")
        if split == "train":
            truth = work.load_arrays("train/truth", ["n_true"])
            if use_lgb:
                keep_lr = (group_rank_desc(a, p_lr) <= cfg["k2"]) | (group_rank_desc(b, p_lr) == 1)
                report_recall(cand["y"][keep_lr], truth["n_true"], a[keep_lr],
                              rec["country"][:meta["n1"]], meta["countries"], log, "rerank LR (for comparison)")
            report_recall(sub["y"], truth["n_true"], sub["a"], rec["country"][:meta["n1"]],
                          meta["countries"], log, "rerank")
            part, part_names = pool_of(work, split, meta, rec["country"])
            if len(part_names) > len(meta["countries"]):
                report_recall(sub["y"], truth["n_true"], sub["a"], part[:meta["n1"]], part_names,
                              log, "rerank by pool")


def step_features(args, work, cfg, n_jobs, log):
    """Pair features, fs_llr (Fellegi-Sunter, supervised fit unless fs_em), then the v5.5 pair and
    candidate-structure features. Also stores each record's name key (work/<split>/keys)."""
    from .fsem import fs_llr, sup_from_json, sup_to_json
    from .pairfeats import (BASE_FEATURES, FULL_FEATURES, REC_FIELDS, V55_CAND, V55_PAIR,
                            V55_REC_FIELDS, candidate_structure, full_features, name_keys,
                            v55_pair_features)
    sup, fits = None, {}
    splits = _splits(args)
    if "train" not in splits:
        sup = sup_from_json(work.load_json("model/fsem.json")["supervised"])
    col = {n: i for i, n in enumerate(FULL_FEATURES)}
    for split in splits:
        t0 = time.time()
        meta = work.load_json(f"{split}/meta.json")
        n1 = meta["n1"]
        rec = work.load_arrays(f"{split}/rec", sorted(set(REC_FIELDS + V55_REC_FIELDS +
                                                        ["country", "pool"])))
        cand = work.load_arrays(f"{split}/cand")
        a, b = cand["a"].astype(np.int64), cand["b"].astype(np.int64)
        X = full_features(rec, cand, extra=len(FULL_FEATURES) - len(BASE_FEATURES))
        log(f"[{split}] pair features {X.shape} ({time.time() - t0:.0f}s)")
        pair_c = rec["country"][:n1].astype(np.int64)[a]
        llr, sup, fits[split] = fs_llr(X[:, :len(BASE_FEATURES)], BASE_FEATURES, pair_c,
                                       meta["countries"], cand.get("y"), sup, cfg, log, split)
        X[:, col["fs_llr"]] = llr
        V = v55_pair_features(rec, a, b)
        i0 = col[V55_PAIR[0]]
        X[:, i0:i0 + len(V55_PAIR)] = V
        del V
        keys = name_keys(rec["n_sorted_p"], rec["n_sorted_b"], rec["pool"].astype(np.int64))
        work.save_arrays(f"{split}/keys", {"keys": keys})
        C = candidate_structure(a, b, keys, n1, X[:, col["jw_sorted"]], X[:, col["atok_cos"]],
                                X[:, col["num_shared"]], X[:, col["addr_empty_b"]])
        i0 = col[V55_CAND[0]]
        X[:, i0:i0 + len(V55_CAND)] = C
        del C, rec
        work.save_arrays(f"{split}/X", {"X": X})
        log(f"[{split}] features {X.shape} ({time.time() - t0:.0f}s)")
    if "train" in splits:
        work.save_json("model/fsem.json", {"supervised": sup_to_json(sup), "fits": fits})


def _save_models(work, prefix, models):
    files = []
    for sfx, m in models:
        f = f"{prefix}{sfx}.txt"
        m.save_model(work.w("model", f))
        files.append(f)
    return files


def step_train(args, work, cfg, n_jobs, log):
    """v5.5 stacking: stage 1 on the pair features, stages 2..n_stages on the previous stage's
    context and group features, fold models on the test path, a holdout scored like test, and
    per-country experts + an unconstrained model blended into the last stage (plan W3, W7, W8)."""
    from .decide import macro_f05
    from .groupfeats import GROUP_REC_FIELDS
    from .models import gain_ranks, s1_folds, select_cols, stage1_names
    from .pairfeats import FULL_FEATURES, V55_FEATURES
    from .stack import apply_blend, blend_weights, cv_stage, mono_vector_for, next_stage
    meta = work.load_json("train/meta.json")
    countries = meta["countries"]
    n1, N = meta["n1"], meta["n1"] + meta["n2"] + meta["n3"]
    X = work.load_arrays("train/X")["X"]
    if X.shape[1] != len(FULL_FEATURES):
        raise ValueError(f"train/X has {X.shape[1]} columns, code expects {len(FULL_FEATURES)}: "
                         "rerun from the features step")
    cand = work.load_arrays("train/cand")
    truth = work.load_arrays("train/truth", ["n_true"])
    a, b, y = cand["a"].astype(np.int64), cand["b"].astype(np.int64), cand["y"].astype(np.int8)
    s1_c = work.load_arrays("train/rec", ["country"])["country"][:n1].astype(np.int64)
    pair_c = s1_c[a]
    hold = _hold(work, cfg)
    log(f"train pairs={len(y)} positives={int(y.sum())} of {int(truth['n_true'].sum())} true; "
        f"holdout S1s={int(hold.sum())} ({cfg.get('holdout_frac', 0)}); test_path={cfg.get('test_path')} "
        f"n_stages={cfg['n_stages']}")
    baseline = _baseline_report(work)
    grec = work.load_arrays("train/rec", GROUP_REC_FIELDS)
    keys = work.load_arrays("train/keys")["keys"]

    names = stage1_names(cfg)
    Xc = select_cols(X, FULL_FEATURES, names)
    del X
    t_stage = time.time()
    p, models, info = cv_stage(Xc, y, a, n1, cfg, n_jobs, mono_vector_for(names, cfg), log,
                               "stage1", hold)
    stages = [{"names": names, "models": _save_models(work, "stage1", models), "info": info}]
    scores = {"p1": p}
    first_models = {1: models[0][1]}
    n_st = int(cfg["n_stages"])
    for level in range(2, n_st + 1):
        t_stage = time.time()
        Xn, namesn, G = next_stage(Xc, names, p, level - 1, a, b, n1, N, grec, keys, cfg)
        if level == 2:
            work.save_arrays("train/G", {"G": G})
        del Xc, G
        Xc, names = Xn, namesn
        seeds = int(cfg.get("last_stage_seeds", 1)) if level == n_st else 1
        p, models, info = cv_stage(Xc, y, a, n1, cfg, n_jobs, mono_vector_for(names, cfg), log,
                                   f"stage{level}", hold, seeds=seeds)
        stages.append({"names": names, "models": _save_models(work, f"stage{level}", models),
                       "info": info})
        scores[f"p{level}"] = p
        first_models[level] = models[0][1]
    stage_min = (time.time() - t_stage) / 60
    # W8: experts and an unconstrained model on the last stage's matrix, blended per country
    stack = {"stages": stages, "n_stages": n_st, "experts": {}, "unconstrained": [], "blend": {},
             "extra_order": []}
    extras = {name: [] for name in countries}
    want_x, want_u = bool(cfg.get("experts")), bool(cfg.get("unconstrained"))
    if (want_x or want_u) and guard(cfg, 1.2 * stage_min * (want_x + want_u), "experts/unconstrained", log):
        if want_x:
            stack["extra_order"].append("expert")
            for c, name in enumerate(countries):
                rm = pair_c == c
                if not (rm & ~hold[a]).any():
                    continue          # no training rows (e.g. a leave-one-country-out country)
                pe, mods, _ = cv_stage(Xc, y, a, n1, cfg, n_jobs, mono_vector_for(names, cfg), log,
                                       f"expert_{name}", hold, row_mask=rm)
                stack["experts"][name] = _save_models(work, f"stage{n_st}x_{name}", mods)
                extras[name].append(pe)
        if want_u:
            stack["extra_order"].append("unconstrained")
            pu, mods, _ = cv_stage(Xc, y, a, n1, cfg, n_jobs, mono_vector_for(names, cfg, True), log,
                                   "unconstrained", hold)
            stack["unconstrained"] = _save_models(work, f"stage{n_st}u", mods)
            for name in countries:
                extras[name].append(pu)
        for c, name in enumerate(countries):
            m = (pair_c == c) & ~hold[a]
            if not m.any() or not extras[name]:
                continue
            parts = np.column_stack([p] + extras[name])
            w, ll = blend_weights(parts, y, m)
            stack["blend"][name] = w
            log(f"  blend {name}: weights {dict(zip(['pooled'] + stack['extra_order'], w))} "
                f"log loss {ll:.5f}")
        pf = apply_blend(p, extras, pair_c, countries, stack["blend"])
    else:
        pf = p
    del Xc
    work.save_json("model/stack.json", stack)
    scores.update({"pf": pf.astype(np.float32), "fold": s1_folds(n1, cfg["n_folds"], cfg["seed"]),
                   "hold": hold})
    if "p2" not in scores:
        scores["p2"] = pf.astype(np.float32)
    work.save_arrays("train/scores", scores)
    last = first_models[n_st]
    imp = last.feature_importance("gain")
    ev = ~hold[a]
    extra = {"stage1_thr05_f05": macro_f05(a, (scores["p1"] >= 0.5) & ev, y, truth["n_true"], ~hold),
             "top_features": [names[i] for i in np.argsort(-imp)[:30]],
             "v55_gain_rank_stage1": gain_ranks(first_models[1], stages[0]["names"], V55_FEATURES),
             "features": {f"stage{k + 1}": len(s["names"]) for k, s in enumerate(stages)},
             "stack": {"n_stages": n_st, "test_path": cfg.get("test_path"),
                       "blend": stack["blend"], "best_iters": [s["info"]["best_iters"] for s in stages]}}
    if baseline:
        extra["baseline"] = baseline
    run_tune(work, cfg, n_jobs, log, extra)
    if baseline and "oof_f05" in baseline:
        rep = work.load_json("model/report.json")
        log(f"  OOF macro F0.5 {rep['oof_f05']:.5f} vs baseline {baseline['oof_f05']:.5f} "
            f"({baseline['from']}): {100 * (rep['oof_f05'] - baseline['oof_f05']):+.3f} points")


def _baseline_report(work):
    """The previous run's report.json in a read-only --work-in folder (v3 for v4), if any: its
    OOF is directly comparable when prepare/block/rerank are reused."""
    import json
    for base in work.ins:
        path = os.path.join(base, "model", "report.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                rep = json.load(f)
            keys = [k for k in rep if k.startswith(("oof_f05", "matches_per_s1", "empty_share"))]
            return {"from": path, **{k: rep[k] for k in keys}}
    return None


def _hm_extra(cfg, X, x_names, G):
    """v4 has-match inputs available under the feature switches: {name: per-pair array}."""
    from .groupfeats import G_COLUMNS, GROUP_FEATURES
    extra = {}
    if cfg["feat_fs"] and "fs_llr" in x_names:
        extra["fs_llr"] = np.asarray(X[:, x_names.index("fs_llr")], np.float32)
    if cfg["feat_group"] and G is not None:
        for n in GROUP_FEATURES:
            extra[n] = G[:, G_COLUMNS.index(n)]
    return extra


def run_tune(work, cfg, n_jobs, log, extra=None):
    """Decision step on saved OOF scores: calibration, has-match model, decision settings; writes
    model/decision.json, model/hasmatch.txt, train/scores/{q,sel,p_has} and model/report.json.
    v5.5 (decision_v55): settings per group (training country, and the France-like pool used for
    France), tuned on the non-holdout S1s; the holdout is then scored with the chosen settings."""
    from .decide import (HM_PAIR_COLS, calibrate, decide_group, decide_v55, fit_calibration,
                         fit_hasmatch, fit_isotonic, apply_isotonic, hm_features, hm_names,
                         macro_f05, take_cols, tune, tune_group)
    from .retrieval import pool_of
    meta = work.load_json("train/meta.json")
    n1, N = meta["n1"], meta["n1"] + meta["n2"] + meta["n3"]
    countries = meta["countries"]
    cand = work.load_arrays("train/cand", ["a", "b", "y"])
    truth = work.load_arrays("train/truth", ["n_true"])
    rec = work.load_arrays("train/rec", ["country", "a_empty"])
    have_pf = work.exists("train/scores", "pf.npy")
    sc = work.load_arrays("train/scores", ["pf" if have_pf else "p2", "fold"])
    p_fin = sc["pf" if have_pf else "p2"]
    a, b, y = cand["a"].astype(np.int64), cand["b"].astype(np.int64), cand["y"].astype(np.int8)
    n_true = truth["n_true"]
    s1_c = rec["country"][:n1].astype(np.int64)
    pair_c = s1_c[a]
    hold = _hold(work, cfg)
    ev = ~hold
    # countries with training S1s; the others (loco_countries) are decided as unseen countries
    fitted = [name for c, name in enumerate(countries) if (ev & (s1_c == c)).any()]
    pev = ev[a]
    part, part_names = pool_of(work, "train", meta, rec["country"])
    s1_p = part[:n1].astype(np.int64)
    pair_p = s1_p[a]

    cal = fit_calibration(p_fin[pev], y[pev], pair_c[pev], countries, cfg["calib_min_pairs"], log)
    cal["by_pool"] = {}
    unseen_map = {}
    if cfg.get("decision_v55"):
        for unseen, pool in _unseen_pools(cfg, meta):
            if pool in part_names:
                m = (pair_p == part_names.index(pool)) & pev
                cal["by_pool"][pool] = fit_isotonic(p_fin[m].astype(np.float64), y[m].astype(np.float64))
                unseen_map[unseen] = f"pool:{pool}"
        if unseen_map:
            log(f"  unseen countries -> pool settings: {unseen_map}")
    q = calibrate(p_fin, pair_c, countries, cal)
    p_has, hm_cols = None, None
    if cfg["has_match"]:
        Xm = work.load_arrays("train/X", mmap=True)["X"]
        x_names = _x_names(Xm.shape[1])
        want = HM_PAIR_COLS + (["fs_llr"] if "fs_llr" in x_names else [])
        C = take_cols(Xm, [x_names.index(n) for n in want])
        G = work.load_arrays("train/G")["G"] if work.exists("train/G") else None
        hm_extra = _hm_extra(cfg, C, want, G)   # not `extra`: that is the report argument
        s1, F = hm_features(a, q, {n: C[:, i] for i, n in enumerate(want)}, rec["a_empty"], hm_extra)
        hm_cols = hm_names(hm_extra)
        del C, G, hm_extra
        log(f"  has-match features ({len(hm_cols)}): {hm_cols}")
        p_has, bst = fit_hasmatch(s1, F, n_true, sc["fold"], n_jobs, cfg["seed"], log, hold=hold)
        bst.save_model(work.w("model", "hasmatch.txt"))

    report = dict(work.load_json("model/report.json")) if (extra is None and
                                                          work.exists("model/report.json")) else {}
    report.update(extra or {})
    if cfg.get("decision_v55"):
        groups = {}
        for c, name in enumerate(countries):
            idx = np.nonzero(pair_c == c)[0]
            if len(idx) and name in fitted:
                groups[name] = tune_group(a[idx], b[idx], q[idx], y[idx], n_true, N,
                                          ev & (s1_c == c), cfg, p_has, log, name)
        fr_q = {}
        for unseen, key in unseen_map.items():
            pool = key[len("pool:"):]
            pi = part_names.index(pool)
            idx = np.nonzero(pair_p == pi)[0]
            fr_q[key] = (idx, apply_isotonic(p_fin[idx], cal["by_pool"][pool]))
            groups[key] = tune_group(a[idx], b[idx], fr_q[key][1], y[idx], n_true, N,
                                     ev & (s1_p == pi), cfg, p_has, log, key)
        groups["*"] = groups[next(iter(unseen_map.values()))] if unseen_map else groups[fitted[0]]
        dec = {"decision_v55": True, "groups": groups, "unseen_map": unseen_map,
               "has_match": any(g.get("has_match") or g.get("gate", 0) > 0 for g in groups.values()),
               "excl": groups["*"]["excl"],
               "excl_by_country": {n: groups[n]["excl"] for n in countries if n in groups}}
        dec["train_countries"] = fitted
        sel, qe = decide_v55(a, b, q, N, dec, pair_c, countries, p_has)
        dec["f05"] = macro_f05(a, sel, y, n_true, ev)
        # France path on the France-like pool: its own settings and the unseen-country floor
        for key, (idx, qp) in fr_q.items():
            pi = part_names.index(key[len("pool:"):])
            sel_f, qe_f = decide_group(a[idx], b[idx], qp, N, groups[key], p_has)
            floor = sel_f & ~(qe_f < cfg["unseen_min_q"])
            for tag, mask in (("oof", ev), ("holdout", hold)):
                m = mask & (s1_p == pi)
                if m.any():
                    report[f"{tag}_f05_france_path"] = macro_f05(a[idx], floor, y[idx], n_true, m)
                    report[f"{tag}_f05_france_path_nofloor"] = macro_f05(a[idx], sel_f, y[idx], n_true, m)
        report["decision"] = {k: {**{kk: g[kk] for kk in ("excl", "has_match", "lam", "min_p", "gate", "f05")},
                                  "post_cal": bool(g.get("post_cal"))} for k, g in groups.items()}
    else:
        dec, sel, qe = tune(a, b, q, y, n_true, N, s1_c, countries, cfg, p_has, log)
        report["decision"] = {k: dec[k] for k in ("excl", "excl_by_country", "has_match")}
    dec["hm_features"] = hm_cols
    dec["calibration"] = cal
    dec["pi_tr"] = {name: float(y[(pair_c == c) & pev].mean()) for c, name in enumerate(countries)
                    if ((pair_c == c) & pev).any()}
    dec["pi_tr"]["*"] = float(y[pev].mean())
    dec.setdefault("train_countries", fitted)
    work.save_json("model/decision.json", dec)
    scores = {"q": q, "sel": sel}
    if p_has is not None:
        scores["p_has"] = p_has
    work.save_arrays("train/scores", scores)

    npred = np.bincount(a[sel], minlength=n1)
    single = n_true == 0
    report.update({
        "oof_f05": macro_f05(a, sel, y, n_true, ev),
        "pair_precision": float(y[sel & pev].mean()) if (sel & pev).any() else 0.0,
        "pair_recall": float(y[sel & pev].sum() / max(n_true[ev].sum(), 1)),
        "candidate_recall_ceiling": float(y[pev].sum() / max(n_true[ev].sum(), 1)),
        "singleton_accuracy": float((npred[single & ev] == 0).mean()) if (single & ev).any() else None})
    if hold.any():
        report["holdout_f05"] = macro_f05(a, sel, y, n_true, hold)
        report["holdout_s1"] = int(hold.sum())
    for c, name in enumerate(countries):
        m = s1_c == c
        report[f"oof_f05_{name}"] = macro_f05(a, sel, y, n_true, m & ev)
        if hold.any():
            report[f"holdout_f05_{name}"] = macro_f05(a, sel, y, n_true, m & hold)
        report[f"matches_per_s1_{name}"] = float(npred[m & ev].mean()) if (m & ev).any() else None
        report[f"empty_share_{name}"] = float((npred[m & ev] == 0).mean()) if (m & ev).any() else None
    for c, name in enumerate(countries):
        if name not in fitted:
            # leave-one-country-out: decided like an unseen test country (its group, the floor)
            m = s1_c == c
            fl = sel & ~((qe < cfg["unseen_min_q"]) & (pair_c == c))
            report[f"loco_f05_{name}"] = macro_f05(a, fl, y, n_true, m)
            report[f"loco_f05_{name}_nofloor"] = macro_f05(a, sel, y, n_true, m)
    if len(part_names) > len(countries):
        for c, name in enumerate(part_names):
            m = s1_p == c
            if m.any():
                if (m & ev).any():
                    report[f"oof_f05_pool_{name}"] = macro_f05(a, sel, y, n_true, m & ev)
                if hold.any():
                    report[f"holdout_f05_pool_{name}"] = macro_f05(a, sel, y, n_true, m & hold)
                if (m & ev).any():
                    report[f"matches_per_s1_pool_{name}"] = float(npred[m & ev].mean())
                    report[f"empty_share_pool_{name}"] = float((npred[m & ev] == 0).mean())
    work.save_json("model/report.json", report)
    for k, v in report.items():
        if k != "decision":
            log(f"  {k}: {v}")
    log(f"  decision: {report['decision']}")


def _x_names(ncols):
    """Column names of a stored X: v5.5's FULL_FEATURES, v4/v5's, or v3's 46."""
    from .pairfeats import CTX_FEATURES, FULL_FEATURES, PAIR_FEATURES, V4_FEATURES
    if ncols == len(FULL_FEATURES):
        return FULL_FEATURES
    if ncols == len(V4_FEATURES):
        return V4_FEATURES
    v3 = PAIR_FEATURES[:PAIR_FEATURES.index("nn_shared")] + CTX_FEATURES
    if ncols == len(v3):
        return v3
    raise ValueError(f"stored X has {ncols} columns; expected {len(FULL_FEATURES)}, "
                     f"{len(V4_FEATURES)} or {len(v3)}")


def _unseen_pools(cfg, meta):
    """[(unseen country, training pool)]: unseen_pool="auto" (or {}) reads the sampling plan, where
    each extra pool names the test country it was shaped like (encode, pools_extra); a dict gives the
    mapping explicitly. Countries are an open set: nothing here names one."""
    up = cfg.get("unseen_pool")
    if isinstance(up, dict) and up:
        return list(up.items())
    return [(pl["like"], name) for name, pl in (meta.get("sampling") or {}).items()
            if pl.get("like") and pl["like"] != pl.get("country")]


def step_tune(args, work, cfg, n_jobs, log):
    run_tune(work, cfg, n_jobs, log)


def _test_context(work):
    meta = work.load_json("test/meta.json")
    n1, N = meta["n1"], meta["n1"] + meta["n2"] + meta["n3"]
    cand = work.load_arrays("test/cand", ["a", "b", "rr_p"])
    rec = work.load_arrays("test/rec", ["country", "ids"])
    a, b = cand["a"].astype(np.int64), cand["b"].astype(np.int64)
    s1_c = rec["country"][:n1].astype(np.int64)
    return meta, n1, N, a, b, cand["rr_p"], s1_c, rec["ids"].astype(str)


def _unseen_codes(meta, dec):
    seen = set(dec["train_countries"])
    return [c for c, name in enumerate(meta["countries"]) if name not in seen]


def _unseen_floor(sel, q, a, s1_c, meta, dec, cfg, log, tag, floor=None):
    """Countries never seen in training get a stricter probability floor (precision first)."""
    unseen_codes = _unseen_codes(meta, dec)
    floor = cfg["unseen_min_q"] if floor is None else floor
    if not unseen_codes or floor <= 0:
        return sel
    unseen = np.isin(s1_c, unseen_codes)[a]
    dropped = sel & unseen & (q < floor)
    log(f"[{tag}] unseen countries {[meta['countries'][c] for c in unseen_codes]}: "
        f"floor {floor} dropped {int(dropped.sum())} matches")
    return sel & ~dropped


def _test_phas(work, cfg, dec, a, q, X, G, n1, n_jobs):
    """Test P(has a true match) per S1 from calibrated q, or None when the decision doesn't use it."""
    import lightgbm as lgb
    from .decide import HM_PAIR_COLS, hm_features, hm_names, predict_hasmatch
    if not dec["has_match"]:
        return None
    names = _x_names(X.shape[1])
    C = {n: X[:, names.index(n)] for n in HM_PAIR_COLS}
    extra = _hm_extra(cfg, X, names, G)
    if dec.get("hm_features") is not None and hm_names(extra) != dec["hm_features"]:
        raise ValueError(f"has-match features {hm_names(extra)} differ from training's "
                         f"{dec['hm_features']}: use the same feat_* settings as train")
    a_empty = work.load_arrays("test/rec", ["a_empty"])["a_empty"]
    s1, F = hm_features(a, q, C, a_empty, extra)
    return predict_hasmatch(lgb.Booster(model_file=work.r("model", "hasmatch.txt")), s1, F, n1,
                            n_jobs)


def main_lambda(cfg, countries):
    """Per-country odds multiplier of the main file (country_lambda; 1 elsewhere)."""
    lam = np.ones(len(countries))
    for name, v in (cfg.get("country_lambda") or {}).items():
        if name in countries:
            lam[countries.index(name)] = float(v)
    return lam


def decide_lam(a, b, q, N, dec, pair_c, s1_c, countries, cfg, p_has, lam_c):
    """decide() with a per-country odds multiplier lam_c (None or all ones: none). The has-match
    probability moves with the same lambda."""
    from .decide import decide, lam_shift
    if lam_c is None or np.all(lam_c == 1.0):
        return decide(a, b, q, N, dec, pair_c, countries, cfg["min_p"], p_has)
    ph = p_has
    if p_has is not None:
        ph = np.where(p_has >= 0, lam_shift(np.clip(p_has, 0, 1), lam_c[s1_c]), -1.0)
    return decide(a, b, q, N, dec, pair_c, countries, cfg["min_p"], ph, lam_c[pair_c])


def _load_boosters(work, files):
    import lightgbm as lgb
    return [(lgb.Booster(model_file=work.r("model", f)), 0) for f in files]


def test_scores(work, cfg, n_jobs, log, X, a, b, n1, N, countries, pair_c, tag="test"):
    """The stack on test pairs: stage 1..n (mean of each stage's saved models, inputs built by
    stack.next_stage exactly as in training), then the expert/unconstrained blend.
    -> (scores dict with p1..pn and pf, stage-1 group features G for the has-match model)."""
    from .groupfeats import GROUP_REC_FIELDS
    from .models import select_cols
    from .pairfeats import FULL_FEATURES
    from .stack import apply_blend, next_stage, predict_models
    st = work.load_json("model/stack.json")
    grec = work.load_arrays("test/rec", GROUP_REC_FIELDS)
    keys = work.load_arrays("test/keys")["keys"]
    names = st["stages"][0]["names"]
    Xc = select_cols(X, FULL_FEATURES, names)
    p = predict_models(_load_boosters(work, st["stages"][0]["models"]), Xc, n_jobs)
    scores, G1 = {"p1": p}, None
    for level in range(2, st["n_stages"] + 1):
        s = st["stages"][level - 1]
        Xn, namesn, G = next_stage(Xc, names, p, level - 1, a, b, n1, N, grec, keys, cfg)
        if namesn != s["names"]:
            raise ValueError(f"stage {level} columns differ from training's: same feat_* settings needed")
        if level == 2:
            G1 = G
        del Xc
        Xc, names = Xn, namesn
        p = predict_models(_load_boosters(work, s["models"]), Xc, n_jobs)
        scores[f"p{level}"] = p
        log(f"[{tag}] stage {level}: mean of {len(s['models'])} model(s)")
    extras = {name: [] for name in countries}
    for kind in st.get("extra_order", []):
        if kind == "unconstrained":
            pu = predict_models(_load_boosters(work, st["unconstrained"]), Xc, n_jobs)
            for name in countries:
                extras[name].append(pu)
        else:
            for c, name in enumerate(countries):
                if name not in st["experts"]:
                    continue
                m = pair_c == c
                pe = np.full(len(a), np.nan, np.float32)
                if m.any():
                    pe[m] = predict_models(_load_boosters(work, st["experts"][name]), Xc[m], n_jobs)
                extras[name].append(pe)
    scores["pf"] = apply_blend(p, extras, pair_c, countries, st.get("blend", {}))
    if "p2" not in scores:
        scores["p2"] = scores["pf"]
    return scores, G1


def step_predict(args, work, cfg, n_jobs, log):
    from .decide import calibrate, calibrate_v55
    from .pairfeats import FULL_FEATURES
    meta, n1, N, a, b, rr_p, s1_c, ids = _test_context(work)
    X = work.load_arrays("test/X")["X"]
    if X.shape[1] != len(FULL_FEATURES):
        raise ValueError(f"test/X has {X.shape[1]} columns, code expects {len(FULL_FEATURES)}")
    countries = meta["countries"]
    pair_c = s1_c[a]
    scores, G = test_scores(work, cfg, n_jobs, log, X, a, b, n1, N, countries, pair_c)
    if G is not None:
        work.save_arrays("test/G", {"G": G})
    dec = work.load_json("model/decision.json")
    if "groups" in dec:
        q = calibrate_v55(scores["pf"], pair_c, countries, dec["calibration"], dec)
        log(f"[test] decision groups per country: "
            f"{ {n: _group_of(dec, n) for n in countries} }")
    else:
        q = calibrate(scores["pf"], pair_c, countries, dec["calibration"])
    scores["q"] = q
    p_has = _test_phas(work, cfg, dec, a, q, X, G, n1, n_jobs)
    if p_has is not None:
        scores["p_has"] = p_has
    del X, G
    lam_c = main_lambda(cfg, countries)
    if not np.all(lam_c == 1.0):
        log(f"[test] main country_lambda: {dict(zip(countries, lam_c.round(4).tolist()))}")
    sel, qe = decide_lam(a, b, q, N, dec, pair_c, s1_c, countries, cfg, p_has, lam_c)
    sel = _unseen_floor(sel, qe, a, s1_c, meta, dec, cfg, log, "test")
    scores["sel"] = sel
    work.save_arrays("test/scores", scores)
    write_outputs(args.out, a, b, sel, qe, rr_p, n1, ids, candidates=True)
    log_counts(sel, a, s1_c, countries, n1, log, "test main")


def _group_of(dec, name):
    from .decide import group_key
    return group_key(dec, name)


def write_outputs(out_dir, a, b, sel, q, rr_p, n1, ids, candidates):
    os.makedirs(out_dir, exist_ok=True)
    s1_ids = ids[:n1].tolist()
    if candidates:
        write_id_lists(os.path.join(out_dir, "candidate_pairs.tsv"),
                       ["source1_entity_id", "candidate_entity_ids"], s1_ids,
                       _lists(a, b, rr_p, n1, ids))
    write_id_lists(os.path.join(out_dir, "matching_results.tsv"),
                   ["source1_entity_id", "matched_entity_ids"], s1_ids,
                   _lists(a[sel], b[sel], q[sel], n1, ids))


def log_counts(sel, a, s1_c, countries, n1, log, tag):
    n_match = np.bincount(a[sel], minlength=n1)
    parts = [f"{name}: {n_match[s1_c == c].mean():.3f}/S1, {(n_match[s1_c == c] == 0).mean():.4f} "
             f"empty" for c, name in enumerate(countries) if (s1_c == c).any()]
    log(f"[{tag}] {int(sel.sum())} matches, {(n_match == 0).mean():.4f} of S1 empty; "
        + "; ".join(parts))


def _lists(a, b, score, n1, ids):
    order = np.lexsort((-score, a))
    a_s, b_s = a[order], b[order]
    starts = np.searchsorted(a_s, np.arange(n1 + 1))
    return [ids[b_s[starts[i]:starts[i + 1]]].tolist() for i in range(n1)]


# ------------------------------------------------------------------ variant files
def _validate(validator, out_dir, test_dir, candidates, log):
    if not validator or not os.path.exists(validator):
        log(f"  validator not found ({validator}); skipped")
        return None
    cmd = [sys.executable, validator, "--matching", os.path.join(out_dir, "matching_results.tsv"),
           "--test-dir", test_dir]
    if candidates:
        cmd += ["--candidate", os.path.join(out_dir, "candidate_pairs.tsv")]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    log(f"  validator {'PASS' if r.returncode == 0 else 'FAIL'}: {' | '.join(tail)}")
    return r.returncode == 0


def step_variants(args, work, cfg, n_jobs, log):
    """Variant files rebuilt from saved test scores; each differs from main in one setting.
    main carries country_lambda; lamX and unseen_lamX multiply it (unseen = every test country absent
    from training, an open set); prior_em replaces it; v5.5 adds unseen_v5 (unseen countries decided
    with pooled calibration and the global settings, as in v5) and unseen_nofloor."""
    from .decide import calibrate, prior_em
    meta, n1, N, a, b, rr_p, s1_c, ids = _test_context(work)
    countries = meta["countries"]
    pair_c = s1_c[a]
    dec = work.load_json("model/decision.json")
    have_ph = work.exists("test/scores", "p_has.npy")
    sc = work.load_arrays("test/scores", ["q", "pf"] + (["p_has"] if have_ph else []))
    q, p_has = sc["q"], sc.get("p_has")
    base = main_lambda(cfg, countries)
    unseen = _unseen_codes(meta, dec)

    # prior_em: a lambda per country from the test match rate (Saerens, Latinne, Decaestecker 2002)
    em_lam = np.ones(len(countries))
    for c, name in enumerate(countries):
        m = pair_c == c
        if not m.any():
            continue
        pi_tr = dec["pi_tr"].get(name, dec["pi_tr"]["*"])
        pi, lam = prior_em(q[m], pi_tr)
        em_lam[c] = lam
        log(f"  prior_em {name}: pi_tr={pi_tr:.4f} pi_test={pi:.4f} lambda={lam:.3f}")
    variants = [("main", base, None)]
    variants += [(f"lam{v:g}", base * float(v), None) for v in cfg["variant_lambdas"]]
    if unseen:
        for v in cfg["variant_fr_lambdas"]:
            lam_c = base.copy()
            lam_c[unseen] *= float(v)
            variants.append((f"unseen_lam{v:g}", lam_c, None))
        if "groups" in dec and cfg.get("variant_fr_v5"):
            variants.append(("unseen_v5", base, "v5"))
        if cfg.get("variant_fr_nofloor"):
            variants.append(("unseen_nofloor", base, "nofloor"))
    variants.append(("prior_em", em_lam, None))

    vdir = os.path.join(args.out, "variants")
    test_dir = os.path.join(args.data, "test")
    rows, main_sets, ok = [], None, True
    main_sel = main_qe = None
    for name, lam_c, mode in variants:
        d, qq, floor = dec, q, None
        if mode == "v5":
            d = {**dec, "unseen_map": {}}          # unseen -> "*" settings
            qq = calibrate(sc["pf"], pair_c, countries, dec["calibration"])   # pooled for unseen
        elif mode == "nofloor":
            floor = 0.0
        sel, qe = decide_lam(a, b, qq, N, d, pair_c, s1_c, countries, cfg, p_has, lam_c)
        sel = _unseen_floor(sel, qe, a, s1_c, meta, dec, cfg, log, name, floor)
        if mode is not None and main_sets is not None:
            # identical to main outside the unseen countries
            um = np.isin(pair_c, unseen)
            sel = np.where(um, sel, main_sel)
            qe = np.where(um, qe, main_qe)
        out_dir = os.path.join(vdir, name)
        write_outputs(out_dir, a, b, sel, qe, rr_p, n1, ids, candidates=False)
        log_counts(sel, a, s1_c, countries, n1, log, f"variant {name}")
        passed = _validate(args.validator, out_dir, test_dir, False, log)
        ok &= passed is not False
        keys = np.sort(a[sel] * N + b[sel])
        n_match = np.bincount(a[sel], minlength=n1)
        if main_sets is None:
            main_sets = (keys, n_match)
            main_sel, main_qe = sel, qe
        diff = _s1_diff(keys, main_sets[0], N, n1)
        for c, cname in enumerate(countries):
            m = s1_c == c
            if m.any():
                rows.append([name, cname, f"{n_match[m].mean():.4f}", f"{(n_match[m] == 0).mean():.4f}",
                             str(int(diff[m].sum())), f"{lam_c[c]:.4f}",
                             {True: "PASS", False: "FAIL", None: "skipped"}[passed], ""])
    write_summary(vdir, rows)
    passed = _validate(args.validator, args.out, test_dir, True, log)
    ok &= passed is not False
    if not ok:
        raise RuntimeError("a variant file failed the official validator (see log)")


def _s1_diff(keys, main_keys, N, n1):
    """Per S1: 1 if its predicted set differs from main's."""
    only = np.setxor1d(keys, main_keys, assume_unique=True)
    d = np.zeros(n1, bool)
    d[(only // N).astype(np.int64)] = True
    return d


SUMMARY_COLS = ["variant", "country", "matches_per_s1", "empty_share", "s1_differs_from_main",
                "lambda", "validator", "flag"]


def write_summary(vdir, rows, append=False):
    """output/variants/summary.tsv; append=True replaces earlier rows of the same variants."""
    os.makedirs(vdir, exist_ok=True)
    path = os.path.join(vdir, "summary.tsv")
    old = []
    if append and os.path.exists(path):
        names = {r[0] for r in rows}
        with open(path, encoding="utf-8") as f:
            old = [line.rstrip("\n").split("\t") for line in list(f)[1:]]
        old = [r + [""] * (len(SUMMARY_COLS) - len(r)) for r in old if r and r[0] not in names]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(SUMMARY_COLS) + "\n")
        for r in old + rows:
            f.write("\t".join(r) + "\n")


# ------------------------------------------------------------------ v5: self-training
def _hold(work, cfg, split="train"):
    """Holdout S1s of a split (v5.5 W7): the random holdout plus every S1 of loco_countries."""
    from .stack import holdout_s1
    meta = work.load_json(f"{split}/meta.json")
    s1_c = work.load_arrays(f"{split}/rec", ["country"])["country"][:meta["n1"]].astype(np.int64)
    return holdout_s1(meta["n1"], cfg, s1_c, meta["countries"])


def rebuild_last(work, cfg, split, X, a, b, n1, N, stage_scores):
    """The last stage's input matrix of a split, rebuilt from saved stage scores p1..p(n-1) with
    stack.next_stage (as in training) -> (matrix, column names)."""
    from .groupfeats import GROUP_REC_FIELDS
    from .models import select_cols
    from .pairfeats import FULL_FEATURES
    from .stack import next_stage
    st = work.load_json("model/stack.json")
    grec = work.load_arrays(f"{split}/rec", GROUP_REC_FIELDS)
    keys = work.load_arrays(f"{split}/keys")["keys"]
    names = st["stages"][0]["names"]
    Xc = select_cols(X, FULL_FEATURES, names)
    for level in range(2, st["n_stages"] + 1):
        Xn, names, _ = next_stage(Xc, names, stage_scores[f"p{level - 1}"], level - 1, a, b, n1, N,
                                  grec, keys, cfg)
        del Xc
        Xc = Xn
    if names != st["stages"][-1]["names"]:
        raise ValueError("rebuilt last-stage columns differ from training's")
    return Xc, names


def _pseudo_labels(a, b, q, sel_main, rows, n1, N, cfg, seed):
    """v5 rules on the target rows: pseudo-positives q >= st_pos_q whose record's runner-up S1 has
    q <= st_runner_up_q and whose S1's main list has <= st_max_list records; pseudo-negatives: those
    S1s' other target candidates with q <= st_neg_q; at most st_max_s1 S1s. -> (pos, neg) masks."""
    from .nbutils import group_top2
    F = np.zeros(len(a), bool)
    F[rows] = True
    q64 = q.astype(np.float64)
    t1, t2, _, _, _ = group_top2(b, np.where(F, q64, -1.0), N)
    runner = np.maximum(np.where(q64 >= t1[b], t2[b], t1[b]), 0.0)
    npred = np.bincount(a[sel_main], minlength=n1)
    pos = F & (q64 >= cfg["st_pos_q"]) & (runner <= cfg["st_runner_up_q"]) & (npred[a] <= cfg["st_max_list"])
    s1s = np.unique(a[pos])
    if len(s1s) > cfg["st_max_s1"]:
        s1s = np.sort(np.random.RandomState(seed).choice(s1s, cfg["st_max_s1"], replace=False))
    in_s1 = np.zeros(n1, bool)
    in_s1[s1s] = True
    pos &= in_s1[a]
    neg = F & in_s1[a] & (q64 <= cfg["st_neg_q"]) & ~pos
    return pos, neg, len(s1s)


def step_selftrain(args, work, cfg, n_jobs, log):
    """Self-training for countries unseen in training (v5.5; the FAQ allows it): pseudo-labels on
    their confident pairs join the last stage's training rows at weight st_weight, one refit (the
    last stage's rounds) rescores their pairs, and the result is decided like main (their group
    settings, calibration and floor). Output: variant unseen_selftrain, identical to main outside the
    unseen countries, flagged DRIFT when more than st_drift of their S1s change list.
    With loco_countries (leave-one-country-out) it runs on those training countries instead, whose
    labels were kept out of every fit, and reports the F0.5 before and after (model/selftrain.json)."""
    from .decide import apply_isotonic, calibrate, decide_group, group_key, macro_f05
    from .models import fit_lgb, lgb_threads, take_rows
    from .stack import mono_vector_for
    if not cfg["self_train"]:
        log("  self_train=false: skipped")
        return
    if not work.exists("model/stack.json"):
        log("  self-training needs the v5.5 stack (model/stack.json): skipped")
        return
    if not guard(cfg, cfg["selftrain_min"], "self-training", log):
        return
    st = work.load_json("model/stack.json")
    dec = work.load_json("model/decision.json")
    loco = [c for c in (cfg.get("loco_countries") or [])]
    tmeta = work.load_json("train/meta.json")
    tn1, tN = tmeta["n1"], tmeta["n1"] + tmeta["n2"] + tmeta["n3"]
    tc = work.load_arrays("train/cand", ["a", "b", "y"])
    ta, tb, ty = tc["a"].astype(np.int64), tc["b"].astype(np.int64), tc["y"].astype(np.int8)
    tsc = work.load_arrays("train/scores")
    hold = _hold(work, cfg)
    Xt, names = rebuild_last(work, cfg, "train", work.load_arrays("train/X")["X"], ta, tb, tn1, tN, tsc)
    fit = ~hold[ta]
    cap = int(cfg["max_train_s1"])
    if cap > 0:
        s1s = np.unique(ta[fit])
        if len(s1s) > cap:
            keep = np.zeros(tn1, bool)
            keep[np.random.RandomState(cfg["seed"] + 5).choice(s1s, cap, replace=False)] = True
            fit &= keep[ta]
    rounds = int(st["stages"][-1]["info"]["rounds"])
    mono = mono_vector_for(names, cfg)
    t0 = time.time()

    def refit(Xp, yp):
        Xall = np.concatenate([take_rows(Xt, fit), Xp])
        yall = np.concatenate([ty[fit], yp])
        wall = np.concatenate([np.ones(int(fit.sum()), np.float32),
                               np.full(len(yp), cfg["st_weight"], np.float32)])
        bst = fit_lgb(Xall, yall, None, None, cfg, lgb_threads(cfg, n_jobs), mono, rounds=rounds,
                      log=log, weight=wall)
        log(f"  refit on {len(yall)} rows ({len(yp)} pseudo) in {(time.time() - t0) / 60:.1f} min")
        return bst

    if loco:
        # ---- leave-one-country-out: the held-out training countries play the unseen country
        s1_c = work.load_arrays("train/rec", ["country"])["country"][:tn1].astype(np.int64)
        codes = [tmeta["countries"].index(c) for c in loco if c in tmeta["countries"]]
        T = np.isin(s1_c[ta], codes)
        rows = np.nonzero(T)[0]
        q, sel_main = tsc["q"], tsc["sel"].astype(bool)
        pos, neg, n_s1 = _pseudo_labels(ta, tb, q, sel_main, rows, tn1, tN, cfg, cfg["seed"] + 11)
        log(f"  [loco {loco}] pseudo-labels: {n_s1} S1s, {int(pos.sum())} positives "
            f"({float(ty[pos].mean()) if pos.any() else 0:.4f} truly positive), {int(neg.sum())} "
            f"negatives ({float(1 - ty[neg].mean()) if neg.any() else 0:.4f} truly negative)")
        if not pos.any():
            return
        lab = pos | neg
        bst = refit(Xt[lab], pos[lab].astype(np.int8))
        p_new = bst.predict(Xt[rows], num_threads=n_jobs).astype(np.float32)
        out = {"loco": loco, "pseudo_s1": n_s1, "pos": int(pos.sum()), "neg": int(neg.sum())}
        truth = work.load_arrays("train/truth", ["n_true"])["n_true"]
        for c in codes:
            name = tmeta["countries"][c]
            m_s1 = s1_c == c
            rc = np.nonzero(s1_c[ta] == c)[0]
            gst = dec["groups"][group_key(dec, name)]
            for tag, p_c in (("before", tsc["pf"][rc]), ("after", p_new[np.searchsorted(rows, rc)])):
                qq = calibrate(p_c, np.zeros(len(rc), np.int64), ["*"], dec["calibration"])
                s, e = decide_group(ta[rc], tb[rc], qq, tN, gst, tsc.get("p_has"))
                for fl in (0.0, cfg["unseen_min_q"]):
                    sf = s & ~(e < fl) if fl > 0 else s
                    out[f"f05_{name}_{tag}_floor{fl:g}"] = macro_f05(ta[rc], sf, ty[rc], truth, m_s1)
        work.save_json("model/selftrain.json", out)
        for k, v in out.items():
            log(f"  loco selftrain {k}: {v}")
        return

    # ---- test: the countries unseen in training
    meta, n1, N, a, b, rr_p, s1_c, ids = _test_context(work)
    countries = meta["countries"]
    unseen = _unseen_codes(meta, dec)
    targets = [c for c in unseen if (cfg.get("self_train_countries") in (None, "auto")
                                     or countries[c] in cfg["self_train_countries"])]
    if not targets:
        log("  no unseen test countries: skipped")
        return
    pair_c = s1_c[a]
    sc = work.load_arrays("test/scores")
    q, sel_main = sc["q"], sc["sel"].astype(bool)
    rows = np.nonzero(np.isin(pair_c, targets))[0]
    pos, neg, n_s1 = _pseudo_labels(a, b, q, sel_main, rows, n1, N, cfg, cfg["seed"] + 11)
    log(f"  pseudo-labels in {[countries[c] for c in targets]}: {n_s1} S1s, {int(pos.sum())} "
        f"positives, {int(neg.sum())} negatives")
    if not pos.any():
        log("  no pseudo-positives: skipped")
        return
    Xte, names_te = rebuild_last(work, cfg, "test", work.load_arrays("test/X")["X"], a, b, n1, N, sc)
    lab = pos | neg
    bst = refit(Xte[lab], pos[lab].astype(np.int8))
    Xt = None
    bst.save_model(work.w("model", "last_selftrain.txt"))
    p_new = sc["pf"].copy()
    p_new[rows] = bst.predict(Xte[rows], num_threads=n_jobs)
    del Xte
    q2 = q.copy()
    for c in targets:
        m = pair_c == c
        key = dec.get("unseen_map", {}).get(countries[c], "")
        iso = dec["calibration"].get("by_pool", {}).get(key[len("pool:"):]) if key.startswith("pool:") else None
        q2[m] = (apply_isotonic(p_new[m], iso) if iso is not None else
                 calibrate(p_new[m], np.zeros(int(m.sum()), np.int64), ["*"], dec["calibration"]))
    p_has = sc.get("p_has")
    sel, qe = decide_lam(a, b, q2, N, dec, pair_c, s1_c, countries, cfg, p_has,
                         main_lambda(cfg, countries))
    sel = _unseen_floor(sel, qe, a, s1_c, meta, dec, cfg, log, "unseen_selftrain")
    F = np.isin(pair_c, targets)
    sel = np.where(F, sel, sel_main)
    keys = np.sort(a[sel] * N + b[sel])
    diff = _s1_diff(keys, np.sort(a[sel_main] * N + b[sel_main]), N, n1)
    tmask = np.isin(s1_c, targets)
    share = float(diff[tmask].mean())
    flag = "DRIFT" if share > cfg["st_drift"] else ""
    log(f"  {share:.3%} of unseen-country S1s changed their list (limit {cfg['st_drift']:.0%})"
        + ("  -> FLAGGED AS DRIFT: do not upload" if flag else ""))
    out_dir = os.path.join(args.out, "variants", "unseen_selftrain")
    write_outputs(out_dir, a, b, sel, np.where(F, qe, q), rr_p, n1, ids, candidates=False)
    if flag:
        with open(os.path.join(out_dir, "DRIFT"), "w") as f:
            f.write(f"{share:.4f} of unseen-country S1s changed their list\n")
    log_counts(sel, a, s1_c, countries, n1, log, "variant unseen_selftrain")
    passed = _validate(args.validator, out_dir, os.path.join(args.data, "test"), False, log)
    n_match = np.bincount(a[sel], minlength=n1)
    rows_s = []
    for c, cname in enumerate(countries):
        mc = s1_c == c
        if mc.any():
            rows_s.append(["unseen_selftrain", cname, f"{n_match[mc].mean():.4f}",
                           f"{(n_match[mc] == 0).mean():.4f}", str(int(diff[mc].sum())), "",
                           {True: "PASS", False: "FAIL", None: "skipped"}[passed], flag])
    write_summary(os.path.join(args.out, "variants"), rows_s, append=True)
    work.save_json("model/selftrain.json", {"s1": int(n_s1), "pos": int(pos.sum()),
                                            "neg": int(neg.sum()), "changed_share": share,
                                            "drift": bool(flag)})
    if passed is False:
        raise RuntimeError("unseen_selftrain failed the official validator (see log)")


def step_diagnose(args, work, cfg, n_jobs, log):
    from .diagnose import run_diagnose
    if not guard(cfg, cfg["diagnose_min"], "diagnostics", log):
        return
    run_diagnose(args, work, cfg, n_jobs, log)


# ------------------------------------------------------------------ v5: checkpoints
CKPT_DIRS = ["model", "logs", "diag", "train/scores", "test/scores"]
CKPT_FILES = ["lexicon.json", "run_config.json", "train/meta.json", "test/meta.json"]


def checkpoint(work, dest, done, log):
    """Copy the small artifacts (models, reports, scores, logs) to dest, skipping unchanged files,
    and write dest/STATUS.json. The big work arrays stay on the scratch disk."""
    n = 0
    for rel in CKPT_DIRS + CKPT_FILES:
        src = os.path.join(work.out, rel)
        files = ([(src, os.path.join(dest, rel))] if os.path.isfile(src) else
                 [(os.path.join(r, f), os.path.join(dest, os.path.relpath(os.path.join(r, f), work.out)))
                  for r, _, fs in os.walk(src) for f in fs] if os.path.isdir(src) else [])
        for s, d in files:
            st = os.stat(s)
            if os.path.exists(d) and os.path.getsize(d) == st.st_size and os.path.getmtime(d) >= st.st_mtime:
                continue
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
            n += 1
    # the notebook runs each step in its own process: keep the steps earlier processes finished
    path = os.path.join(dest, "STATUS.json")
    prev = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                prev = json.load(f).get("done", [])
        except (OSError, ValueError):
            prev = []
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"done": prev + [s for s in done if s not in prev],
                   "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "minutes_since_start": round((time.time() - T0) / 60, 1)}, f, indent=1)
    log(f"  checkpoint: {n} files -> {dest}")


# ------------------------------------------------------------------ sample for smoke tests
def make_sample(data, out, n_s1, seed, log):
    rng = np.random.RandomState(seed)

    def write(df, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(df.columns) + "\n")
            for row in df.itertuples(index=False):
                f.write("\t".join(row) + "\n")

    gt = read_ground_truth(os.path.join(data, "train", "train_ground_truth.tsv"))
    gt = gt.iloc[rng.choice(len(gt), min(n_s1, len(gt)), replace=False)]
    s1 = read_source(os.path.join(data, "train", "train_source1.tsv"))
    write(s1[s1.entity_id.isin(set(gt.source1_entity_id))], os.path.join(out, "train", "train_source1.tsv"))
    matched = {x for s in gt.matched_entity_ids if s for x in s.split(",")}
    for k in (2, 3):
        df = read_source(os.path.join(data, "train", f"train_source{k}.tsv"))
        extra = rng.rand(len(df)) < (0.25 * len(matched) / 2) / len(df)  # distractors
        write(df[df.entity_id.isin(matched) | extra], os.path.join(out, "train", f"train_source{k}.tsv"))
    write(gt, os.path.join(out, "train", "train_ground_truth.tsv"))
    # test: one shared fraction for S1, S2 and S3 so per-country densities stay realistic
    # (the match_test sampling reads them)
    tests = [read_source(os.path.join(data, "test", f"test_source{k}.tsv")) for k in (1, 2, 3)]
    frac = min(1.0, (n_s1 / 4) / len(tests[0]))
    for k, df in zip((1, 2, 3), tests):
        write(df[rng.rand(len(df)) < frac], os.path.join(out, "test", f"test_source{k}.tsv"))
    log(f"sample written to {out} (test fraction {frac:.4f})")


# ------------------------------------------------------------------ v5: submission package
PACKAGE_CODE = ["run.py", "requirements.txt", "README.md"]
PACKAGE_DIRS = {"src": (".py",), "tools": (".py", ".sh"), "kaggle": (".ipynb",),
                "sagemaker": (".py", ".md")}   # the README links the SageMaker guide


def package(args, log):
    """<team>_submission.zip in the challenge README's layout: output/ (the chosen matching file and
    candidate_pairs.tsv), code/business_entity_resolution/ (this code), Documentation_template.md
    (the filled methodology), plus MANIFEST.json with sha256 of every file."""
    import hashlib
    import zipfile
    run = args.run or args.out
    matching = args.matching or os.path.join(run, "matching_results.tsv")
    cands = os.path.join(run, "candidate_pairs.tsv")
    doc = args.doc or os.path.join(CODE_ROOT, "docs", "Documentation.md")
    zpath = args.out if args.out.endswith(".zip") else os.path.join(args.out, f"{args.team}_submission.zip")
    for p in (matching, cands, doc):
        if not os.path.exists(p):
            raise FileNotFoundError(f"package: {p} is missing")
    if args.data:
        tmp =os.path.join(os.path.dirname(os.path.abspath(zpath)), "_pkg_check")
        os.makedirs(tmp, exist_ok=True)
        shutil.copy(matching, os.path.join(tmp, "matching_results.tsv"))
        shutil.copy(cands, os.path.join(tmp, "candidate_pairs.tsv"))
        ok = _validate(args.validator, tmp, os.path.join(args.data, "test"), True, log)
        shutil.rmtree(tmp, ignore_errors=True)
        if ok is False:
            raise RuntimeError("package: the files fail the official validator")
    entries = [(matching, "output/matching_results.tsv"), (cands, "output/candidate_pairs.tsv"),
               (doc, "Documentation_template.md")]
    base = "code/business_entity_resolution"
    entries += [(os.path.join(CODE_ROOT, f), f"{base}/{f}") for f in PACKAGE_CODE]
    for d, exts in PACKAGE_DIRS.items():
        full = os.path.join(CODE_ROOT, d)
        if os.path.isdir(full):
            entries += [(os.path.join(full, f), f"{base}/{d}/{f}") for f in sorted(os.listdir(full))
                        if f.endswith(exts)]
    if args.work and os.path.exists(os.path.join(args.work, "run_config.json")):
        entries.append((os.path.join(args.work, "run_config.json"), f"{base}/run_config.json"))
    manifest = {}
    os.makedirs(os.path.dirname(os.path.abspath(zpath)), exist_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in entries:
            with open(src, "rb") as f:
                manifest[arc] = hashlib.sha256(f.read()).hexdigest()
            z.write(src, arc)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=1))
    log(f"package: {zpath} ({len(entries)} files, {os.path.getsize(zpath) / 2**20:.0f} MB); "
        f"matching sha256 {manifest['output/matching_results.tsv'][:16]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["make-sample", "all", "tune", "package"] + STEPS)
    ap.add_argument("--data", help="folder with train/ and test/ TSVs (package: optional, validates)")
    ap.add_argument("--work", default="work", help="artifact folder (written)")
    ap.add_argument("--work-in", nargs="*", default=[], help="extra read-only artifact folders")
    ap.add_argument("--out", default="output", help="submission output folder (package: the zip)")
    ap.add_argument("--from", dest="from_step", choices=STEPS, help="all: first step to run")
    ap.add_argument("--to", dest="to_step", choices=STEPS, help="all: last step to run")
    ap.add_argument("--validator", default=os.path.join(CODE_ROOT, "tools", "validate_submission.py"),
                    help="official validator run on every output file")
    ap.add_argument("--n-s1", type=int, default=20000, help="make-sample: S1 entities to keep")
    ap.add_argument("--profile", help="named override set from config.PROFILES, applied before --set")
    ap.add_argument("--checkpoint", help="copy models/reports/scores/logs here after every step")
    ap.add_argument("--run", help="package: output folder of the chosen run")
    ap.add_argument("--matching", help="package: matching file to ship (default: <run>/matching_results.tsv)")
    ap.add_argument("--doc", help="package: filled documentation (default: docs/Documentation.md)")
    ap.add_argument("--team", default="team", help="package: team name for the zip file name")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--splits", default="train,test",
                    help="prepare/block/rerank/features: splits to (re)compute, e.g. test")
    args = ap.parse_args(argv)
    if args.step != "package" and not args.data:
        ap.error("--data is required")
    cfg = load_config(args.set, args.profile)
    cfg["_t0"] = T0                      # run start, for the fit-level time guard (models.fit_lgb)
    n_jobs = cfg["n_jobs"] or os.cpu_count()
    if args.step == "package":
        package(args, print)
        return
    import numba
    numba.set_num_threads(min(n_jobs, numba.config.NUMBA_NUM_THREADS))
    work = Work(args.work, args.work_in)
    log = _logger(work, args.step)
    log(f"step={args.step} n_jobs={n_jobs} profile={args.profile} config overrides={args.set}")
    if args.step == "make-sample":
        make_sample(args.data, args.out, args.n_s1, cfg["seed"], log)
        return
    work.save_json("run_config.json", {"profile": args.profile, "overrides": args.set, "config": cfg,
                                       "n_jobs": n_jobs})
    if args.step == "all":
        lo = STEPS.index(args.from_step) if args.from_step else 0
        hi = STEPS.index(args.to_step) if args.to_step else len(STEPS) - 1
        steps = STEPS[lo:hi + 1]
    else:
        steps = [args.step]
    funcs = {"prepare": step_prepare, "block": step_block, "rerank": step_rerank,
             "features": step_features, "train": step_train, "predict": step_predict,
             "variants": step_variants, "selftrain": step_selftrain, "diagnose": step_diagnose,
             "tune": step_tune}
    done = []
    for s in steps:
        t0 = time.time()
        log(f"=== {s} === ({(time.time() - T0) / 60:.0f} min since start, "
            f"{minutes_left(cfg):.0f} min left for optional work)")
        funcs[s](args, work, cfg, n_jobs, log)
        done.append(s)
        log(f"=== {s} done in {time.time() - t0:.0f}s ===")
        if args.checkpoint:
            checkpoint(work, args.checkpoint, done, log)


if __name__ == "__main__":
    sys.exit(main())
