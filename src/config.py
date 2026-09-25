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
    # rarity weights of the model features: per country (v3) or over the whole split (v1/v2)
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
    "fs_max_iter": 200,         # EM iterations
    "fs_tol": 1e-7,             # EM stop: change in mean log-likelihood per pair
    # GBDT (v4 capacity: 800k S1s per fit, 255 leaves, 2000 rounds)
    "n_folds": 3,
    "max_train_s1": 800000,     # S1 entities per model fit (caps training rows)
    "lgb_rounds": 2000,
    "lgb_lr": 0.08,
    "lgb_leaves": 255,
    "lgb_min_leaf": 200,
    "lgb_early_stop": 100,
    # decision layer
    "min_p": 0.001,
    "unseen_min_q": 0.7,        # stricter floor for countries absent from training (France)
    "calib_min_pairs": 1000000,  # per-country isotonic calibration above this many OOF pairs
    "delta_grid": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],  # hard-exclusivity margins
    "soft_excl": True,          # include soft exclusivity in the grid
    "has_match": True,          # include the per-S1 has-match model in the grid
    # variant files (rebuilt from saved test scores)
    "variant_lambdas": [0.6, 1.6],
    "variant_fr_lambdas": [0.5, 2.0],
    # diagnostics
    "dump_errors": 500,         # error samples per type
}


def load_config(overrides):
    cfg = dict(DEFAULTS)
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
