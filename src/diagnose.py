"""Step `diagnose`: error samples and counts on the OOF predictions, and a train/test shift report.

diag/errors_<type>.tsv   samples per error type, stratified by country
diag/error_counts.json   counts per type and country, and the F0.5 each type costs
diag/shift_report.tsv    per country and feature: train vs test percentiles and a KS statistic
"""
import json
import os

import numpy as np
import pandas as pd

from .decide import f05_from_counts
from .groupfeats import G_COLUMNS
from .io_utils import read_source
from .models import CTX2_FEATURES, p_context
from .pairfeats import FULL_FEATURES, PAIR_FEATURES, REC_FIELDS, pair_features

TYPES = ["block_miss", "rerank_miss", "dropped_true", "false_match", "singleton_fp"]
KEY_FEATS = ["jw_sorted", "ntok_cos", "atok_cos", "tri_addr", "num_shared", "num_conflict",
             "state_cat", "legal_cat", "name_freq_a", "n_cand_b", "p1_other_b",
             "nn_shared", "nn_conflict", "acro", "gw_a_only", "gw_b_only", "fs_llr"]
Q_BANDS = [0.1, 0.3, 0.5, 0.7]
PCTS = [5, 25, 50, 75, 95]
SHIFT_ROWS = 1_000_000   # rows sampled per country and split for the shift report


def _q_band(q):
    labels = ["<0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", ">=0.7"]
    return np.array(labels, dtype=object)[np.searchsorted(Q_BANDS, q, side="right")]


def _strs(p, buf, rows):
    return [bytes(buf[p[r]:p[r + 1]]).decode("ascii", "replace") for r in rows]


def _nums(p, d, rows):
    return [" ".join(str(x) for x in d[p[r]:p[r + 1]]) for r in rows]


def _stratified(idx, country, k, rng):
    """Up to k of idx, split evenly across countries (leftover quota goes to the others)."""
    if len(idx) <= k:
        return idx
    groups = [idx[country[idx] == c] for c in np.unique(country[idx])]
    groups.sort(key=len)
    out, left = [], k
    for i, g in enumerate(groups):
        take = min(len(g), left // (len(groups) - i))
        out.append(rng.choice(g, take, replace=False))
        left -= take
    return np.sort(np.concatenate(out))


def run_diagnose(args, work, cfg, n_jobs, log):
    rng = np.random.RandomState(cfg["seed"])
    os.makedirs(work.w("diag", "x"), exist_ok=True)
    meta = work.load_json("train/meta.json")
    n1, N = meta["n1"], meta["n1"] + meta["n2"] + meta["n3"]
    countries = meta["countries"]
    truth = work.load_arrays("train/truth")
    pool_true, n_true = truth["pool_true"].astype(np.int64), truth["n_true"]
    cand = work.load_arrays("train/cand", ["a", "b", "y"])
    a, b, y = cand["a"].astype(np.int64), cand["b"].astype(np.int64), cand["y"].astype(bool)
    sc = work.load_arrays("train/scores", ["p1", "p2", "q", "sel"])
    sel = sc["sel"].astype(bool)
    rec = work.load_arrays("train/rec", sorted(set(REC_FIELDS) | {"country", "ids", "nu_p", "nu_d"}))
    s1_c = rec["country"][:n1].astype(np.int64)

    # ---- which true pairs survive blocking and re-ranking (each pool record has <= 1 true S1)
    raw = work.load_arrays("train/cand_raw", ["a", "b", "score", "n_cand_b"])
    ra, rb = raw["a"].astype(np.int64), raw["b"].astype(np.int64)
    in_raw = np.zeros(N, bool)
    in_raw[rb[pool_true[rb] == ra]] = True
    in_cand = np.zeros(N, bool)
    in_cand[b[y]] = True
    true_b = np.nonzero(pool_true >= 0)[0]
    bm_b = true_b[~in_raw[true_b]]
    rm_b = true_b[in_raw[true_b] & ~in_cand[true_b]]
    types = {
        "block_miss": ("pool", bm_b),
        "rerank_miss": ("pool", rm_b),
        "dropped_true": ("cand", np.nonzero(y & ~sel)[0]),
        "false_match": ("cand", np.nonzero(sel & ~y & (n_true[a] > 0))[0]),
        "singleton_fp": ("cand", np.nonzero(sel & (n_true[a] == 0))[0]),
    }

    # ---- counts and the F0.5 each type costs (fix only that type in the OOF predictions)
    tp = np.bincount(a[sel & y], minlength=n1)
    npred = np.bincount(a[sel], minlength=n1)
    base = f05_from_counts(tp, npred, n_true)
    counts = {"oof_f05": float(base.mean()), "types": {}}
    for t, (kind, rows) in types.items():
        s1 = pool_true[rows] if kind == "pool" else a[rows]
        per = np.bincount(s1, minlength=n1)
        if t in ("block_miss", "rerank_miss", "dropped_true"):
            fixed = f05_from_counts(tp + per, npred + per, n_true)
        else:
            fixed = f05_from_counts(tp, npred - per, n_true)
        gain = fixed - base
        entry = {"count": int(len(rows)), "f05_lost": float(gain.mean())}
        for c, name in enumerate(countries):
            m = s1_c == c
            entry[f"count_{name}"] = int(per[m].sum())
            entry[f"f05_lost_{name}"] = float(gain[m].mean()) if m.any() else None
        counts["types"][t] = entry
        log(f"  {t}: {entry}")
    _compare_baseline(work, counts, log)
    with open(work.w("diag", "error_counts.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=1)

    # ---- samples
    p1_other_b = p_context(a, b, sc["p1"], n1, N)[:, CTX2_FEATURES.index("p1_other_b")]
    rank_a = _rank_within(a, sc["p2"])
    top_row = _top_row(a, sc["p2"], n1)                        # best candidate per S1
    best_true_row = _top_row(a[y], sc["p2"][y], n1, rows=np.nonzero(y)[0])
    Xm = work.load_arrays("train/X", mmap=True)["X"]
    raw_pos = _raw_positions(ra, rb, N)
    samples = {}
    for t, (kind, rows) in types.items():
        if kind == "pool":
            pb = _stratified(rows, rec["country"], cfg["dump_errors"], rng)
            pa = pool_true[pb]
            crow = np.full(len(pb), -1)
        else:
            crow = _stratified(rows, s1_c[a], cfg["dump_errors"], rng)
            pa, pb = a[crow], b[crow]
        samples[t] = (pa, pb, crow)
    need = set()
    for pa, pb, crow in samples.values():
        need.update(rec["ids"][pa].astype(str))
        need.update(rec["ids"][pb].astype(str))
        extra = np.concatenate([top_row[pa], best_true_row[pa]])
        need.update(rec["ids"][b[extra[extra >= 0]]].astype(str))
    rawrec = _raw_records(args.data, need)
    for t, (pa, pb, crow) in samples.items():
        df = _sample_frame(t, pa, pb, crow, rec, meta, rawrec, sc, a, b, y, rank_a, top_row,
                           best_true_row, Xm, p1_other_b, raw_pos, raw)
        df.to_csv(work.w("diag", f"errors_{t}.tsv"), sep="\t", index=False)
        log(f"  wrote diag/errors_{t}.tsv ({len(df)} rows)")
    del Xm
    shift_report(work, cfg, log)


def _compare_baseline(work, counts, log):
    """Error counts against the previous run's (a read-only --work-in folder: v3 for v4)."""
    for base in work.ins:
        path = os.path.join(base, "diag", "error_counts.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            counts["baseline"] = {"from": path, "oof_f05": old.get("oof_f05"),
                                  "types": old.get("types", {})}
            for t, e in counts["types"].items():
                o = old.get("types", {}).get(t)
                if o:
                    log(f"  vs baseline {t}: count {o['count']} -> {e['count']}, "
                        f"f05_lost {o['f05_lost']:.5f} -> {e['f05_lost']:.5f}")
            return


def _rank_within(a, p):
    from .nbutils import group_rank_desc
    return group_rank_desc(a, p.astype(np.float64))


def _top_row(a, p, n1, rows=None):
    """Row (into the full candidate arrays) of each S1's highest-p candidate, -1 if none."""
    rows = np.arange(len(a)) if rows is None else rows
    out = np.full(n1, -1, np.int64)
    if len(a):
        order = np.lexsort((-p, a))
        first = np.r_[True, a[order][1:] != a[order][:-1]]
        out[a[order][first]] = rows[order][first]
    return out


def _raw_positions(ra, rb, N):
    keys = ra * N + rb
    order = np.argsort(keys)
    return keys[order], order


def _raw_records(data, ids):
    out = {}
    for k in (1, 2, 3):
        df = read_source(os.path.join(data, "train", f"train_source{k}.tsv"))
        df = df[df.entity_id.isin(ids)]
        out.update(zip(df.entity_id, zip(df.business_name, df.business_address)))
    return out


def _sample_frame(t, pa, pb, crow, rec, meta, rawrec, sc, a, b, y, rank_a, top_row, best_true_row,
                  Xm, p1_other_b, raw_pos, raw):
    N = len(rec["country"])
    ids = rec["ids"].astype(str)
    states = meta["states"]

    def state(r):
        return [states[s] if s >= 0 else "" for s in rec["state"][r]]
    df = pd.DataFrame({
        "type": t, "country": [meta["countries"][c] for c in rec["country"][pa]],
        "s1_id": ids[pa], "pool_id": ids[pb], "pool_src": rec["src"][pb],
        "s1_name": [rawrec.get(i, ("", ""))[0] for i in ids[pa]],
        "pool_name": [rawrec.get(i, ("", ""))[0] for i in ids[pb]],
        "s1_addr": [rawrec.get(i, ("", ""))[1] for i in ids[pa]],
        "pool_addr": [rawrec.get(i, ("", ""))[1] for i in ids[pb]],
        "s1_norm_name": _strs(rec["n_core_p"], rec["n_core_b"], pa),
        "pool_norm_name": _strs(rec["n_core_p"], rec["n_core_b"], pb),
        "s1_norm_addr": _strs(rec["a_sorted_p"], rec["a_sorted_b"], pa),
        "pool_norm_addr": _strs(rec["a_sorted_p"], rec["a_sorted_b"], pb),
        "s1_nums": _nums(rec["nu_p"], rec["nu_d"], pa), "pool_nums": _nums(rec["nu_p"], rec["nu_d"], pb),
        "s1_state": state(pa), "pool_state": state(pb),
    })
    in_c = crow >= 0
    cr = np.where(in_c, crow, 0)
    for col, arr in (("p1", sc["p1"]), ("p2", sc["p2"]), ("q", sc["q"])):
        df[col] = np.where(in_c, arr[cr], np.nan)
    df["rank_in_s1"] = np.where(in_c, rank_a[cr], -1)
    if t == "dropped_true":
        df["q_band"] = _q_band(df["q"].to_numpy())
    # key features: candidate rows from X, other pairs computed here (no retrieval context)
    feats = np.full((len(pa), len(KEY_FEATS)), np.nan, np.float32)
    if in_c.any():
        rows = cr[in_c]
        order = np.argsort(rows)
        Xr = np.empty((len(rows), Xm.shape[1]), np.float32)
        Xr[order] = Xm[rows[order]]
        for j, f in enumerate(KEY_FEATS):
            if f in FULL_FEATURES:
                feats[in_c, j] = Xr[:, FULL_FEATURES.index(f)]
        feats[in_c, KEY_FEATS.index("p1_other_b")] = p1_other_b[rows]
    if (~in_c).any():
        P = pair_features(rec, pa[~in_c], pb[~in_c])
        for j, f in enumerate(KEY_FEATS):
            if f in PAIR_FEATURES:
                feats[~in_c, j] = P[:, PAIR_FEATURES.index(f)]
        # rerank misses were retrieved: their n_cand_b and retrieval score come from cand_raw
        keys, order = raw_pos
        k = pa[~in_c] * N + pb[~in_c]
        pos = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
        hit = keys[pos] == k
        ncb = np.full(len(k), np.nan, np.float32)
        ncb[hit] = raw["n_cand_b"][order[pos[hit]]]
        feats[~in_c, KEY_FEATS.index("n_cand_b")] = ncb
        rs = np.full(len(pa), np.nan, np.float32)
        rs[np.nonzero(~in_c)[0][hit]] = raw["score"][order[pos[hit]]]
        df["retrieval_score"] = rs
    for j, f in enumerate(KEY_FEATS):
        df[f] = feats[:, j]
    # context: the S1's top candidate for misses, its best true candidate for false matches
    other = top_row[pa] if t in ("block_miss", "rerank_miss", "dropped_true") else best_true_row[pa]
    label = "top_cand" if t in ("block_miss", "rerank_miss", "dropped_true") else "best_true_cand"
    ok = other >= 0
    ob = np.where(ok, b[np.maximum(other, 0)], 0)
    df[f"{label}_id"] = np.where(ok, ids[ob], "")
    df[f"{label}_name"] = [rawrec.get(ids[x], ("", ""))[0] if o else "" for x, o in zip(ob, ok)]
    df[f"{label}_addr"] = [rawrec.get(ids[x], ("", ""))[1] if o else "" for x, o in zip(ob, ok)]
    df[f"{label}_p2"] = np.where(ok, sc["p2"][np.maximum(other, 0)], np.nan)
    if label == "top_cand":
        df["top_cand_is_true"] = np.where(ok, y[np.maximum(other, 0)], False)
    return df


# ------------------------------------------------------------------ shift report
def _ks(x, z):
    x, z = np.sort(x[~np.isnan(x)]), np.sort(z[~np.isnan(z)])
    if not len(x) or not len(z):
        return np.nan
    v = np.concatenate([x, z])
    return float(np.max(np.abs(np.searchsorted(x, v, "right") / len(x)
                               - np.searchsorted(z, v, "right") / len(z))))


def _split_values(work, split, p1, names, rng):
    """Per country: sampled rows of the requested features (pair features + stage-2 context)."""
    meta = work.load_json(f"{split}/meta.json")
    n1, N = meta["n1"], meta["n1"] + meta["n2"] + meta["n3"]
    cand = work.load_arrays(f"{split}/cand", ["a", "b"])
    a, b = cand["a"].astype(np.int64), cand["b"].astype(np.int64)
    s1_c = work.load_arrays(f"{split}/rec", ["country"])["country"][:n1].astype(np.int64)
    ctx = p_context(a, b, p1, n1, N)
    Xm = work.load_arrays(f"{split}/X", mmap=True)["X"]
    Gm = work.load_arrays(f"{split}/G", mmap=True)["G"] if work.exists(f"{split}/G") else None
    out = {}
    for c, name in enumerate(meta["countries"]):
        rows = np.nonzero(s1_c[a] == c)[0]
        if not len(rows):
            continue
        if len(rows) > SHIFT_ROWS:
            rows = np.sort(rng.choice(rows, SHIFT_ROWS, replace=False))
        Xr = Xm[rows]
        cols = {}
        for f in names:
            if f in FULL_FEATURES:
                cols[f] = Xr[:, FULL_FEATURES.index(f)].astype(np.float64)
            elif f in CTX2_FEATURES:
                cols[f] = ctx[rows, CTX2_FEATURES.index(f)].astype(np.float64)
            elif f in G_COLUMNS and Gm is not None:
                cols[f] = Gm[rows, G_COLUMNS.index(f)].astype(np.float64)
        out[name] = cols
    return out


def shift_report(work, cfg, log):
    rng = np.random.RandomState(cfg["seed"] + 7)
    report = work.load_json("model/report.json")
    names = list(dict.fromkeys(report.get("top_features", [])[:20] +
                               ["name_freq_a", "n_cand_b", "p1_n_b", "state_cat"]))
    tr = _split_values(work, "train", work.load_arrays("train/scores", ["p1"])["p1"], names, rng)
    te = _split_values(work, "test", work.load_arrays("test/scores", ["p1"])["p1"], names, rng)
    pooled = {f: np.concatenate([tr[c][f] for c in tr]) for f in names if all(f in tr[c] for c in tr)}
    rows = []
    for country, cols in te.items():
        ref_name = country if country in tr else "pooled"
        ref = tr.get(country, pooled)
        for f in names:
            if f not in cols or f not in ref:
                continue
            x, z = ref[f], cols[f]
            row = {"country": country, "train_ref": ref_name, "feature": f}
            for p in PCTS:
                row[f"train_p{p}"] = float(np.nanpercentile(x, p)) if np.isfinite(x).any() else np.nan
            for p in PCTS:
                row[f"test_p{p}"] = float(np.nanpercentile(z, p)) if np.isfinite(z).any() else np.nan
            row["ks"] = _ks(x, z)
            row["train_nan"], row["test_nan"] = float(np.isnan(x).mean()), float(np.isnan(z).mean())
            rows.append(row)
        if "state_cat" in cols and "state_cat" in ref:
            for k, lab in enumerate(["both_empty", "same", "one_empty", "differ"]):
                rows.append({"country": country, "train_ref": ref_name,
                             "feature": f"state_cat={lab}",
                             "train_share": float((ref["state_cat"] == k).mean()),
                             "test_share": float((cols["state_cat"] == k).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(work.w("diag", "shift_report.tsv"), sep="\t", index=False, float_format="%.5g")
    med = df.dropna(subset=["ks"]).sort_values("ks", ascending=False).head(10)
    for r in med.itertuples():
        log(f"  shift {r.country:>7} {r.feature:<14} KS={r.ks:.3f} median train "
            f"{r.train_p50:.4g} test {r.test_p50:.4g}")
    log("  wrote diag/shift_report.tsv")
