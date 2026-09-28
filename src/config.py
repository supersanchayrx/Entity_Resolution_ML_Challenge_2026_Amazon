"""Default configuration. Every key can be overridden from the CLI with --set key=value."""
import json

DEFAULTS = {
    # compute
    "n_jobs": 0,                # 0 = all cores
    "seed": 42,
    # training-data realism. match_test: per country, size the pool like test's and drop S1s
    # until S2/S3-per-S1 matches test (v3). legacy: the global keep/drop below (v1/v2)
    "train_sampling": "match_test",
    "train_size_cap": 1.0,      # match_test: scales the kept share down if memory is short
    # legacy (and countries with no test counterpart): drop this share of train S1s so their
    # S2/S3 records become distractors whose true S1 is absent
    "train_drop_s1": 0.19,
    # legacy memory-lean training: use only this fraction of train S1 (pool thinned alike)
    "train_keep_s1": 1.0,
    # v5: extra test-like training pools cut from one country's unused S1s, e.g.
    # [{"name": "us_fr", "from": "us", "like": "france"}]: a US pool with France's test size and
    # density. `pool` drives blocking partitions, IDF groups, name counts and competing-S1 features;
    # `country` keeps driving parsing, states and decision settings. On test, pool = country.
    "pools_extra": [],
    # rarity weights of the model features: per pool (= per country unless pools_extra; v3) or
    # over the whole split (v1/v2)
    "idf_scope": "country",
    # comma list of text fixes to disable (see normalize.FIXES), or "all" for v2 parsing
    "text_off": "",
    # retrieval (bucket creation)
    "k1": 100,                  # forward top-K per S1 from the inverted index
    "rev_k": 5,                 # reverse top-K S1s per S2/S3 record
    "chunk_rows": 1000,         # S1 rows per sparse matmul chunk
    "prefix_m": 48,             # probe only the m highest-weight features of each S1 (prefix filtering)
    "cap_ntok": 20000,          # skip features whose pool document frequency exceeds the cap
    "cap_tri": 5000,
    "cap_atok": 20000,
    "cap_num": 20000,
    "cap_numx": 5000,
    "cap_nnum": 5000,
    "cap_span": 2000,
    "w_ntok": 1.0,              # feature-type multipliers on top of IDF
    "w_tri": 0.35,
    "w_atok": 0.8,
    "w_num": 0.8,
    "w_numx": 1.0,
    "w_nnum": 1.0,
    "w_span": 1.0,
    # re-ranker
    "k2": 25,                   # candidates kept per S1 after the mini re-ranker (= candidate_pairs.tsv)
    "rerank_train_s1": 300000,  # S1 entities sampled to fit the re-ranker
    # lexicon mining
    "lex_pairs": 400000,        # ground-truth pairs sampled for lexicon mining
    "lex_min_count": 40,
    # v4 features (switches for ablations; all on by default)
    "feat_fs": True,            # stage 1: Fellegi-Sunter/EM log-likelihood ratio fs_llr
    "feat_group": True,         # stage 2: agreement with the S1's other likely matches
    "feat_source": True,        # stage 2: source-aware competitor scores
    "g_min_p": 0.5,             # likely set: a's other candidates with p1 >= this ...
    "g_top": 6,                 # ... at most this many, by p1
    "fs_em": False,             # False: every split and country uses the supervised fit (the one the
                                # model trains on). ec2-v4 ran EM per country on test: India/France
                                # converged (lambda 0.39/0.30) while training fell back, and LB fell 1.2 pts
    "fs_max_iter": 200,         # EM iterations
    "fs_tol": 1e-7,             # EM stop: change in mean log-likelihood per pair
    # GBDT (v4 capacity: 800k S1s per fit, 255 leaves, 2000 rounds)
    "n_folds": 3,
    "max_train_s1": 800000,     # S1 entities per model fit (caps training rows); 0 = all
    "lgb_rounds": 2000,
    "lgb_lr": 0.08,
    "lgb_leaves": 255,
    "lgb_min_leaf": 200,
    "lgb_early_stop": 100,
    "lgb_deterministic": False,  # v5: reproducible models (needs the same lgb_threads on rerun)
    "lgb_threads": 0,           # LightGBM threads per fit; 0 = n_jobs
    "cv_parallel": False,       # v5: fit the CV folds at once, each with lgb_threads / n_folds
    "stage2_seeds": 1,          # v5: final stage-2 model = mean of this many seeds (stage 1 keeps 1)
    # decision layer
    "min_p": 0.001,
    # v5: odds multiplier per country applied to main (e.g. {"france": 0.5}), where the leaderboard
    # showed a gain; variants multiply on top of it
    "country_lambda": {},
    "unseen_min_q": 0.7,        # stricter floor for countries absent from training (France)
    "calib_min_pairs": 1000000,  # per-country isotonic calibration above this many OOF pairs
    "delta_grid": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],  # hard-exclusivity margins
    "soft_excl": True,          # include soft exclusivity in the grid
    "has_match": True,          # include the per-S1 has-match model in the grid
    # variant files (rebuilt from saved test scores)
    "variant_lambdas": [0.6, 1.6],
    "variant_fr_lambdas": [0.5, 2.0],
    # self-training on unlabeled test records (the FAQ allows it): variant unseen_selftrain (v5.5)
    "self_train": False,
    "self_train_countries": "auto",   # "auto" = every test country absent from training (open set)
    "st_pos_q": 0.98,           # pseudo-positive: q >= this ...
    "st_runner_up_q": 0.1,      # ... the record's runner-up S1 has q <= this ...
    "st_max_list": 6,           # ... and the S1's predicted list has at most this many records
    "st_neg_q": 0.02,           # pseudo-negatives: the same S1s' other candidates with q <= this
    "st_max_s1": 150000,
    "st_weight": 0.3,
    "st_drift": 0.15,           # flag the variant if more than this share of the S1s change list
    # v5 time guard: optional steps (extra seeds, self-training, diagnostics) are skipped when
    # less than their estimate plus the reserve is left before the deadline (minutes from
    # ER_T0, the notebook start, else the process start)
    "deadline_min": 450,
    "reserve_min": 40,          # kept for predict + variants + outputs
    "selftrain_min": 40,
    "diagnose_min": 10,
    # diagnostics
    "dump_errors": 500,         # error samples per type
    # ---- v5.5. Defaults reproduce v5; profiles v55* switch them on.
    "feat_v55": False,          # W1/W2: v5.5 pair + candidate-structure features and g_twin
    "rerank_model": "lr",       # W6: "lgb" = LightGBM re-ranker (keeps the top k2 by its score)
    "rerank_rounds": 200,
    "holdout_frac": 0.0,        # W7: share of training S1s kept out of every fit, scored like test
    "test_path": "final",       # W3: "folds" = test scores are the mean of the fold models
    "n_stages": 2,              # W3: 3 adds a stage built from stage-2 scores
    "last_stage_seeds": 1,      # seeds per fold model of the last stage (test_path=folds)
    "experts": False,           # W8: per-training-country models of the last stage, blended
    "unconstrained": False,     # W8: a last-stage model without monotone constraints, blended
    "decision_v55": False,      # W4/W5: per-group decisions (lam, post-calibration, min_p, gate)
    "lam_grid": [0.6, 0.7, 0.8, 0.9, 1.12, 1.25, 1.4, 1.6],
    "post_cal": True,
    "min_p_grid": [0.01, 0.05],
    "hm_gate_grid": [0.05, 0.1, 0.2, 0.3],
    # unseen test country -> training pool whose decision settings it uses: "auto" = the pool that
    # pools_extra shaped like it (countries are an open set; no names are hard-coded)
    "unseen_pool": "auto",
    "variant_fr_v5": True,      # variant unseen_v5: unseen countries decided as in v5
    "variant_fr_nofloor": True,  # variant unseen_nofloor: unseen countries without unseen_min_q
    # leave-one-country-out evaluation: these training countries' S1s join the holdout and are
    # decided like an unseen country (report loco_f05_*; self-training reports before/after)
    "loco_countries": [],
}

# v5 section 3: the recall push (k1 150, caps x2 = v2's values, rev_k 10, prefix_m 64, k2 30)
V5_RECALL = {"k1": 150, "rev_k": 10, "prefix_m": 64, "k2": 30,
             "cap_ntok": 40000, "cap_tri": 10000, "cap_atok": 40000, "cap_num": 40000,
             "cap_numx": 10000, "cap_nnum": 10000, "cap_span": 4000}
US_FR = [{"name": "us_fr", "from": "us", "like": "france"}]
_V5 = {**V5_RECALL, "pools_extra": US_FR, "max_train_s1": 0, "lgb_lr": 0.04, "lgb_leaves": 255,
       "lgb_min_leaf": 400, "lgb_rounds": 4000, "lgb_early_stop": 150, "lgb_deterministic": True,
       "cv_parallel": True, "stage2_seeds": 3}
# named override sets, applied before --set (v5 plan sections 7 and 10). v4 = the defaults.
PROFILES = {
    "v4": {},
    "v5": _V5,                                                       # >= 150 GB, >= 64 cores
    "v5_fewcores": {**_V5, "lgb_lr": 0.06, "stage2_seeds": 1},       # >= 150 GB, < 64 cores
    "v5_midmem": {**_V5, "pools_extra": [], "max_train_s1": 1200000},  # 100-150 GB
    "v5_lite": {"k1": 120, "k2": 28, "max_train_s1": 800000, "lgb_lr": 0.06,  # EC2 / < 100 GB
                "lgb_leaves": 255, "lgb_deterministic": True},
}
# v5.5: every v5 profile plus the v5.5 switches (plan section 3). The seeds move from the final model
# to the fold models of the last stage (test_path=folds fits no final model).
V55 = {"pools_extra": "auto", "feat_v55": True, "rerank_model": "lgb", "holdout_frac": 0.05, "test_path": "folds",
       "n_stages": 3, "experts": True, "unconstrained": True, "decision_v55": True,
       "stage2_seeds": 1, "self_train": True}
PROFILES.update({
    "v55": {**_V5, **V55, "last_stage_seeds": 2},
    "v55_fewcores": {**PROFILES["v5_fewcores"], **V55},
    "v55_midmem": {**PROFILES["v5_midmem"], **V55, "pools_extra": [], "experts": False,
                   "unconstrained": False},
    "v55_lite": {**PROFILES["v5_lite"], **V55, "experts": False, "unconstrained": False},
})


def pick_profile(ram_gb, cores, version="v55"):
    """Hardware -> profile name (v5 plan section 7); version "v55" (default) or "v5"."""
    if ram_gb >= 150:
        name = "v5" if cores >= 64 else "v5_fewcores"
    elif ram_gb >= 100:
        name = "v5_midmem"
    else:
        name = "v5_lite"
    return name.replace("v5", version, 1)


def load_config(overrides, profile=None):
    cfg = dict(DEFAULTS)
    if profile:
        if profile not in PROFILES:
            raise KeyError(f"unknown profile: {profile} (known: {sorted(PROFILES)})")
        cfg.update(PROFILES[profile])
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if key not in cfg:
            raise KeyError(f"unknown config key: {key}")
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        cfg[key] = val
    return cfg
