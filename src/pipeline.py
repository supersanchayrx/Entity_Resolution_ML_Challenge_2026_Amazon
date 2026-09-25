"""CLI entry point: python run.py <step> --data DATA --work WORK --out OUT [--set key=value ...]

Steps: make-sample | prepare | block | rerank | features | train | predict | variants | diagnose
       | all (prepare .. diagnose; --from / --to limit the range) | tune (re-tune the decision step
       from saved OOF scores, minutes)
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

from .config import load_config
from .io_utils import Work, read_ground_truth, read_source, write_id_lists

STEPS = ["prepare", "block", "rerank", "features", "train", "predict", "variants", "diagnose"]
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    for split in ("train", "test"):
        prepare_split(split, args.data, work, cfg, n_jobs, log)


def step_block(args, work, cfg, n_jobs, log):
    from .retrieval import retrieve_split
    for split in ("train", "test"):
        retrieve_split(split, work, cfg, n_jobs, log)


def step_rerank(args, work, cfg, n_jobs, log):
    from .models import LogReg, rerank_design
    from .nbutils import group_rank_desc
    from .pairfeats import cheap_features
    from .retrieval import report_recall
    rec_names = ["n_sorted_p", "n_sorted_b", "nt_p", "nt_d", "at_p", "at_d", "nu_p", "nu_d",
                 "n_idf", "a_idf", "state", "country"]
    lr = None
    for split in ("train", "test"):
        rec = work.load_arrays(f"{split}/rec", rec_names)
        cand = work.load_arrays(f"{split}/cand_raw")
        X = cheap_features(rec, cand)
        a, b = cand["a"].astype(np.int64), cand["b"].astype(np.int64)
        if split == "train":
            rng = np.random.RandomState(cfg["seed"])
            s1s = np.unique(a)
            if len(s1s) > cfg["rerank_train_s1"]:
                s1s = rng.choice(s1s, cfg["rerank_train_s1"], replace=False)
            m = np.zeros(a.max() + 1, bool)
            m[s1s] = True
            m = m[a]
            lr = LogReg().fit(rerank_design(X[m]), cand["y"][m].astype(np.float64))
            work.save_json("model/rerank.json", lr.to_dict())
        elif lr is None:
            lr = LogReg.from_dict(work.load_json("model/rerank.json"))
        step = 20_000_000
        p = np.concatenate([lr.predict(rerank_design(X[i:i + step])) for i in range(0, len(X), step)])
        del X
        keep = (group_rank_desc(a, p) <= cfg["k2"]) | (group_rank_desc(b, p) == 1)
        sub = {k: v[keep] for k, v in cand.items()}
        sub["rr_p"] = p[keep].astype(np.float32)
        work.save_arrays(f"{split}/cand", sub)
        meta = work.load_json(f"{split}/meta.json")
        log(f"[{split}] rerank kept {keep.sum()} of {len(keep)} pairs "
            f"({keep.sum() / meta['n1']:.1f} per S1)")
        if split == "train":
            truth = work.load_arrays("train/truth", ["n_true"])
            report_recall(sub["y"], truth["n_true"], sub["a"], rec["country"][:meta["n1"]],
                          meta["countries"], log, "rerank")


def step_features(args, work, cfg, n_jobs, log):
    from .pairfeats import REC_FIELDS, full_features
    for split in ("train", "test"):
        t0 = time.time()
        rec = work.load_arrays(f"{split}/rec", REC_FIELDS)
        cand = work.load_arrays(f"{split}/cand")
        X = full_features(rec, cand)
        work.save_arrays(f"{split}/X", {"X": X})
        log(f"[{split}] features {X.shape} ({time.time() - t0:.0f}s)")


def step_train(args, work, cfg, n_jobs, log):
    from .decide import macro_f05
    from .models import CTX2_FEATURES, CTX2_MONOTONE, cv_lgb, monotone_vector, p_context, s1_folds
    from .pairfeats import FULL_FEATURES, MONOTONE
    meta = work.load_json("train/meta.json")
    n1, N = meta["n1"], meta["n1"] + meta["n2"] + meta["n3"]
    X = work.load_arrays("train/X")["X"]
    cand = work.load_arrays("train/cand")
    truth = work.load_arrays("train/truth", ["n_true"])
    a, b, y = cand["a"].astype(np.int64), cand["b"].astype(np.int64), cand["y"].astype(np.int8)
    log(f"train pairs={len(y)} positives={int(y.sum())} of {int(truth['n_true'].sum())} true")

    oof1, m1 = cv_lgb(X, y, a, n1, cfg, n_jobs, monotone_vector(FULL_FEATURES, MONOTONE), log,
                      "stage1")
    m1.save_model(work.w("model", "stage1.txt"))
    X2 = np.hstack([X, p_context(a, b, oof1, n1, N)])
    del X
    names2 = FULL_FEATURES + CTX2_FEATURES
    mono2 = monotone_vector(names2, {**MONOTONE, **CTX2_MONOTONE})
    oof2, m2 = cv_lgb(X2, y, a, n1, cfg, n_jobs, mono2, log, "stage2")
    m2.save_model(work.w("model", "stage2.txt"))
    del X2
    # saved scores: decision-step experiments (tune, v4) start from here in minutes
    work.save_arrays("train/scores", {"p1": oof1.astype(np.float32), "p2": oof2.astype(np.float32),
                                      "fold": s1_folds(n1, cfg["n_folds"], cfg["seed"])})
    imp = m2.feature_importance("gain")
    extra = {"stage1_thr05_f05": macro_f05(a, oof1 >= 0.5, y, truth["n_true"]),
             "top_features": [names2[i] for i in np.argsort(-imp)[:20]]}
    run_tune(work, cfg, n_jobs, log, extra)


def run_tune(work, cfg, n_jobs, log, extra=None):
    """Decision step on saved OOF scores: calibration, has-match model, grid; writes
    model/decision.json, model/hasmatch.txt, train/scores/{q,sel,p_has} and model/report.json."""
    from .decide import (HM_PAIR_COLS, calibrate, fit_calibration, fit_hasmatch, hm_features,
                         macro_f05, take_cols, tune)
    from .pairfeats import FULL_FEATURES
    meta = work.load_json("train/meta.json")
    n1, N = meta["n1"], meta["n1"] + meta["n2"] + meta["n3"]
    countries = meta["countries"]
    cand = work.load_arrays("train/cand", ["a", "b", "y"])
    truth = work.load_arrays("train/truth", ["n_true"])
    rec = work.load_arrays("train/rec", ["country", "a_empty"])
    sc = work.load_arrays("train/scores", ["p2", "fold"])
    a, b, y = cand["a"].astype(np.int64), cand["b"].astype(np.int64), cand["y"].astype(np.int8)
    n_true = truth["n_true"]
    s1_c = rec["country"][:n1].astype(np.int64)
    pair_c = s1_c[a]

    cal = fit_calibration(sc["p2"], y, pair_c, countries, cfg["calib_min_pairs"], log)
    q = calibrate(sc["p2"], pair_c, countries, cal)
    p_has = None
    if cfg["has_match"]:
        Xm = work.load_arrays("train/X", mmap=True)["X"]
        C = take_cols(Xm, [FULL_FEATURES.index(n) for n in HM_PAIR_COLS])
        s1, F = hm_features(a, q, {n: C[:, i] for i, n in enumerate(HM_PAIR_COLS)}, rec["a_empty"])
        del C
        p_has, bst = fit_hasmatch(s1, F, n_true, sc["fold"], n_jobs, cfg["seed"], log)
        bst.save_model(work.w("model", "hasmatch.txt"))
    dec, sel, qe = tune(a, b, q, y, n_true, N, s1_c, countries, cfg, p_has, log)
    dec["calibration"] = cal
    dec["pi_tr"] = {name: float(y[pair_c == c].mean()) for c, name in enumerate(countries)
                    if (pair_c == c).any()}
    dec["pi_tr"]["*"] = float(y.mean())
    dec["train_countries"] = countries
    work.save_json("model/decision.json", dec)
    scores = {"q": q, "sel": sel}
    if p_has is not None:
        scores["p_has"] = p_has
    work.save_arrays("train/scores", scores)

    report = dict(work.load_json("model/report.json")) if (extra is None and
                                                          work.exists("model/report.json")) else {}
    report.update(extra or {})
    report.update({
        "oof_f05": dec["f05"], "decision": {k: dec[k] for k in ("excl", "excl_by_country",
                                                                "has_match")},
        "pair_precision": float(y[sel].mean()) if sel.any() else 0.0,
        "pair_recall": float(y[sel].sum() / n_true.sum()),
        "candidate_recall_ceiling": float(y.sum() / n_true.sum())})
    npred = np.bincount(a[sel], minlength=n1)
    single = n_true == 0
    report["singleton_accuracy"] = float((npred[single] == 0).mean()) if single.any() else None
    for c, name in enumerate(countries):
        report[f"oof_f05_{name}"] = macro_f05(a, sel, y, n_true, s1_c == c)
        m = s1_c == c
        report[f"matches_per_s1_{name}"] = float(npred[m].mean()) if m.any() else None
        report[f"empty_share_{name}"] = float((npred[m] == 0).mean()) if m.any() else None
    work.save_json("model/report.json", report)
    for k, v in report.items():
        log(f"  {k}: {v}")


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


def _unseen_floor(sel, q, a, s1_c, meta, dec, cfg, log, tag):
    """Countries never seen in training get a stricter probability floor (precision first)."""
    seen = set(dec["train_countries"])
    unseen_codes = [c for c, name in enumerate(meta["countries"]) if name not in seen]
    if not unseen_codes:
        return sel
    unseen = np.isin(s1_c, unseen_codes)[a]
    dropped = sel & unseen & (q < cfg["unseen_min_q"])
    log(f"[{tag}] unseen countries {[meta['countries'][c] for c in unseen_codes]}: "
        f"floor {cfg['unseen_min_q']} dropped {int(dropped.sum())} matches")
    return sel & ~dropped


def step_predict(args, work, cfg, n_jobs, log):
    import lightgbm as lgb
    from .decide import HM_PAIR_COLS, calibrate, decide, hm_features, predict_hasmatch
    from .models import p_context
    from .pairfeats import FULL_FEATURES
    meta, n1, N, a, b, rr_p, s1_c, ids = _test_context(work)
    X = work.load_arrays("test/X")["X"]
    m1 = lgb.Booster(model_file=work.r("model", "stage1.txt"))
    m2 = lgb.Booster(model_file=work.r("model", "stage2.txt"))
    p1 = m1.predict(X, num_threads=n_jobs).astype(np.float32)
    p2 = m2.predict(np.hstack([X, p_context(a, b, p1, n1, N)]),
                    num_threads=n_jobs).astype(np.float32)
    dec = work.load_json("model/decision.json")
    countries = meta["countries"]
    pair_c = s1_c[a]
    q = calibrate(p2, pair_c, countries, dec["calibration"])
    scores = {"p1": p1, "p2": p2, "q": q}
    p_has = None
    if dec["has_match"]:
        C = {n: X[:, FULL_FEATURES.index(n)] for n in HM_PAIR_COLS}
        a_empty = work.load_arrays("test/rec", ["a_empty"])["a_empty"]
        s1, F = hm_features(a, q, C, a_empty)
        p_has = predict_hasmatch(lgb.Booster(model_file=work.r("model", "hasmatch.txt")), s1, F,
                                 n1, n_jobs)
        scores["p_has"] = p_has
    del X
    work.save_arrays("test/scores", scores)
    sel, qe = decide(a, b, q, N, dec, pair_c, countries, cfg["min_p"], p_has)
    sel = _unseen_floor(sel, qe, a, s1_c, meta, dec, cfg, log, "test")
    write_outputs(args.out, a, b, sel, qe, rr_p, n1, ids, candidates=True)
    log_counts(sel, a, s1_c, countries, n1, log, "test main")


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
    """Variant files rebuilt from saved test scores; each differs from main in one setting."""
    from .decide import decide, lam_shift, prior_em
    meta, n1, N, a, b, rr_p, s1_c, ids = _test_context(work)
    countries = meta["countries"]
    pair_c = s1_c[a]
    dec = work.load_json("model/decision.json")
    sc = work.load_arrays("test/scores")
    q, p_has = sc["q"], sc.get("p_has")
    s1_lam_of = {}

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
    fr = [c for c, name in enumerate(countries) if name == "france"]
    variants = [("main", None)]
    variants += [(f"lam{v:g}", np.full(len(countries), float(v))) for v in cfg["variant_lambdas"]]
    for v in cfg["variant_fr_lambdas"]:
        lam_c = np.ones(len(countries))
        lam_c[fr] = float(v)
        variants.append((f"fr_lam{v:g}", lam_c))
    variants.append(("prior_em", em_lam))

    vdir = os.path.join(args.out, "variants")
    test_dir = os.path.join(args.data, "test")
    rows, main_sets, ok = [], None, True
    for name, lam_c in variants:
        lam = None if lam_c is None else lam_c[pair_c]
        ph = p_has
        if lam_c is not None and p_has is not None:
            ph = np.where(p_has >= 0, lam_shift(np.clip(p_has, 0, 1), lam_c[s1_c]), -1.0)
        sel, qe = decide(a, b, q, N, dec, pair_c, countries, cfg["min_p"], ph, lam)
        sel = _unseen_floor(sel, qe, a, s1_c, meta, dec, cfg, log, name)
        out_dir = os.path.join(vdir, name)
        write_outputs(out_dir, a, b, sel, qe, rr_p, n1, ids, candidates=False)
        log_counts(sel, a, s1_c, countries, n1, log, f"variant {name}")
        passed = _validate(args.validator, out_dir, test_dir, False, log)
        ok &= passed is not False
        # per-S1 match sets as sorted (a, b) keys, to count S1s whose list differs from main
        keys = np.sort(a[sel] * N + b[sel])
        n_match = np.bincount(a[sel], minlength=n1)
        if main_sets is None:
            main_sets = (keys, n_match)
        diff = _s1_diff(keys, main_sets[0], N, n1)
        for c, cname in enumerate(countries):
            m = s1_c == c
            if m.any():
                rows.append([name, cname, f"{n_match[m].mean():.4f}", f"{(n_match[m] == 0).mean():.4f}",
                             str(int(diff[m].sum())), "" if lam_c is None else f"{lam_c[c]:.4f}",
                             {True: "PASS", False: "FAIL", None: "skipped"}[passed]])
    os.makedirs(vdir, exist_ok=True)
    with open(os.path.join(vdir, "summary.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("variant\tcountry\tmatches_per_s1\tempty_share\ts1_differs_from_main\tlambda\t"
                "validator\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
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


def step_diagnose(args, work, cfg, n_jobs, log):
    from .diagnose import run_diagnose
    run_diagnose(args, work, cfg, n_jobs, log)


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["make-sample", "all", "tune"] + STEPS)
    ap.add_argument("--data", required=True, help="folder with train/ and test/ TSVs")
    ap.add_argument("--work", default="work", help="artifact folder (written)")
    ap.add_argument("--work-in", nargs="*", default=[], help="extra read-only artifact folders")
    ap.add_argument("--out", default="output", help="submission output folder")
    ap.add_argument("--from", dest="from_step", choices=STEPS, help="all: first step to run")
    ap.add_argument("--to", dest="to_step", choices=STEPS, help="all: last step to run")
    ap.add_argument("--validator", default=os.path.join(CODE_ROOT, "tools", "validate_submission.py"),
                    help="official validator run on every output file")
    ap.add_argument("--n-s1", type=int, default=20000, help="make-sample: S1 entities to keep")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args(argv)
    cfg = load_config(args.set)
    n_jobs = cfg["n_jobs"] or os.cpu_count()
    import numba
    numba.set_num_threads(min(n_jobs, numba.config.NUMBA_NUM_THREADS))
    work = Work(args.work, args.work_in)
    log = _logger(work, args.step)
    log(f"step={args.step} n_jobs={n_jobs} config overrides={args.set}")
    if args.step == "make-sample":
        make_sample(args.data, args.out, args.n_s1, cfg["seed"], log)
        return
    if args.step == "all":
        lo = STEPS.index(args.from_step) if args.from_step else 0
        hi = STEPS.index(args.to_step) if args.to_step else len(STEPS) - 1
        steps = STEPS[lo:hi + 1]
    else:
        steps = [args.step]
    funcs = {"prepare": step_prepare, "block": step_block, "rerank": step_rerank,
             "features": step_features, "train": step_train, "predict": step_predict,
             "variants": step_variants, "diagnose": step_diagnose, "tune": step_tune}
    for s in steps:
        t0 = time.time()
        log(f"=== {s} ===")
        funcs[s](args, work, cfg, n_jobs, log)
        log(f"=== {s} done in {time.time() - t0:.0f}s ===")


if __name__ == "__main__":
    sys.exit(main())
