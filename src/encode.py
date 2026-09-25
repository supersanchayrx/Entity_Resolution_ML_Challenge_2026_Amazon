"""Stage `prepare`: parse every record once and store compact integer/byte arrays.

Record index space per split: S1 records are [0, n1), the pool (S2 then S3) is [n1, N).
"""
import csv
import itertools
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import normalize as nz
from .io_utils import read_ground_truth, read_source
from .lexlearn import learn_lexicon
from .nbutils import csr_sort_unique

CHUNK = 20000
STR_FIELDS = {"n_core": "core", "n_sorted": "sorted", "n_concat": "concat", "n_alt": "alt",
              "a_sorted": "a_sorted"}


def _parse_chunk(args):
    names, addrs, countries = args
    cols = {k: [] for k in ("core", "sorted", "concat", "alt", "legal", "tokens", "phon",
                            "is_domain", "is_indic", "n_nums", "gw", "ini", "a_sorted", "a_tokens",
                            "nums", "state", "masked", "a_empty")}
    for n, a, c in zip(names, addrs, countries):
        pn = nz.parse_name(n, c)
        pa = nz.parse_addr(a, c)
        for k in ("core", "sorted", "concat", "alt", "legal", "tokens", "phon", "is_domain",
                  "is_indic", "gw", "ini"):
            cols[k].append(pn[k])
        cols["n_nums"].append(pn["nums"])
        cols["a_sorted"].append(pa["sorted"])
        cols["a_tokens"].append(pa["tokens"])
        cols["nums"].append(pa["nums"])
        cols["state"].append(pa["state"])
        cols["masked"].append(pa["masked"])
        cols["a_empty"].append(pa["empty"])
    return cols


def parse_frame(df, lex, n_jobs):
    tasks = [(df.business_name.values[i:i + CHUNK], df.business_address.values[i:i + CHUNK],
              df.country.values[i:i + CHUNK]) for i in range(0, len(df), CHUNK)]
    cols = None
    with Pool(n_jobs, initializer=nz.set_lexicon, initargs=(lex,)) as pool:
        for part in pool.imap(_parse_chunk, tasks, chunksize=1):  # stream: no second full copy
            if cols is None:
                cols = {k: [] for k in part}
            for k, v in part.items():
                cols[k].extend(v)
    return cols


def bytes_csr(strings):
    enc = [s.encode("ascii", "replace") for s in strings]
    lens = np.fromiter((len(e) for e in enc), np.int64, len(enc))
    ptr = np.zeros(len(enc) + 1, np.int64)
    np.cumsum(lens, out=ptr[1:])
    buf = np.frombuffer(b"".join(enc), np.uint8).copy() if ptr[-1] else np.zeros(1, np.uint8)
    return ptr, buf


def token_csr(lists, group=None):
    """Lists of strings -> (indptr, sorted-unique int32 ids, vocab list, group of each id).

    With `group` (one small int per list, e.g. the country code) the ids are per (group, token):
    the same word in two countries gets two ids. Vocab strings stay unprefixed."""
    lens = np.fromiter((len(x) for x in lists), np.int64, len(lists))
    ptr = np.zeros(len(lists) + 1, np.int64)
    np.cumsum(lens, out=ptr[1:])
    flat = pd.Series(list(itertools.chain.from_iterable(lists)), dtype=object)
    codes, uniques = pd.factorize(flat)
    uniques = list(uniques)
    id_group = None
    if group is not None and len(codes):
        ng = int(group.max()) + 1
        pair = codes.astype(np.int64) * ng + np.repeat(np.asarray(group, np.int64), lens)
        codes, keys = pd.factorize(pair)
        keys = np.asarray(keys, np.int64)
        uniques = [uniques[t] for t in keys // ng]
        id_group = (keys % ng).astype(np.int32)
    codes = codes.astype(np.int32) if len(codes) else np.zeros(0, np.int32)
    ptr, codes = csr_sort_unique(ptr, codes) if len(codes) else (ptr, codes)
    return ptr, codes, uniques, id_group


def num_csr(lists):
    lens = np.fromiter((len(x) for x in lists), np.int64, len(lists))
    ptr = np.zeros(len(lists) + 1, np.int64)
    np.cumsum(lens, out=ptr[1:])
    data = np.fromiter(itertools.chain.from_iterable(lists), np.int64, int(ptr[-1]))
    return ptr, data


def idf_from_csr(ptr, ids, n_vocab, n_docs):
    df = np.bincount(ids, minlength=n_vocab).astype(np.float64)
    return np.log((n_docs + 1.0) / (df + 1.0)).astype(np.float32) + np.float32(1.0)


def idf_grouped(ids, id_group, docs_per_group):
    """IDF per country: log((N_c + 1) / (df + 1)) + 1, N_c = records of the token's country."""
    df = np.bincount(ids, minlength=len(id_group)).astype(np.float64)
    n_c = np.asarray(docs_per_group, np.float64)[id_group]
    return np.log((n_c + 1.0) / (df + 1.0)).astype(np.float32) + np.float32(1.0)


def codes_with_empty(values):
    codes, uniques = pd.factorize(pd.Series(values, dtype=object))
    uniques = list(uniques)
    codes = codes.astype(np.int32)
    if "" in uniques:
        codes[codes == uniques.index("")] = -1
    return codes, uniques


def _ckey(values):
    return pd.Series(values, dtype=object).str.strip().str.lower().to_numpy()


def test_counts(data_dir):
    """Per country: (test S1 count, test S2+S3 count), from the test files' country column."""
    d = os.path.join(data_dir, "test")
    counts = {}
    for k in (1, 2, 3):
        col = pd.read_csv(os.path.join(d, f"test_source{k}.tsv"), sep="\t", dtype=str,
                          usecols=["country"], keep_default_na=False, na_filter=False,
                          quoting=csv.QUOTE_NONE, encoding="utf-8")["country"]
        for c, n in pd.Series(_ckey(col.values)).value_counts().items():
            counts.setdefault(c, [0, 0])[0 if k == 1 else 1] += int(n)
    return counts


def sample_like_test(srcs, gt, te_counts, cfg, log=print):
    """Shape each country's training S1s and pool like test's, in pool size and S2/S3 per S1.

    k = min(1, cap * m_te / |P_c|): keep each S1 with probability k, with all its matches, and each
    distractor with probability k (pool size like test). d = max(0, 1 - D_tr / D_te): then drop each
    kept S1 with probability d; its matches stay as distractors (density like test). The matches of
    S1s not kept are never kept. Countries without test records: k = cap, d = train_drop_s1."""
    s1, s2, s3 = srcs
    cap, seed = float(cfg["train_size_cap"]), cfg["seed"]
    c1 = _ckey(s1.country.values)
    pool = pd.concat([s2[["entity_id", "country"]], s3[["entity_id", "country"]]], ignore_index=True)
    cp = _ckey(pool.country.values)
    has = gt.matched_entity_ids != ""
    ex = gt.loc[has, ["source1_entity_id"]].assign(
        b=gt.matched_entity_ids[has].str.split(",")).explode("b")
    s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
    b_owner = pd.Series(ex.source1_entity_id.map(s1_pos).values, index=ex.b.values)
    b_owner = b_owner[~b_owner.index.duplicated()]
    owner = pool.entity_id.map(b_owner).to_numpy(np.float64)
    owned = ~np.isnan(owner)
    owner = np.where(owned, owner, -1).astype(np.int64)

    k_s1, d_s1, k_pool = np.zeros(len(s1)), np.zeros(len(s1)), np.zeros(len(pool))
    plan = {}
    for c in sorted(set(c1) | set(cp)):
        m1, mp = c1 == c, cp == c
        n1_tr, m_tr = int(m1.sum()), int(mp.sum())
        if c in te_counts and te_counts[c][0] > 0 and n1_tr > 0 and m_tr > 0:
            n1_te, m_te = te_counts[c]
            d_tr, d_te = m_tr / n1_tr, m_te / n1_te
            k = min(1.0, cap * m_te / m_tr)
            d = max(0.0, 1.0 - d_tr / d_te)
        else:
            d_tr = m_tr / max(n1_tr, 1)
            d_te = None
            k, d = min(1.0, cap), float(cfg["train_drop_s1"])
        k_s1[m1], d_s1[m1], k_pool[mp] = k, d, k
        plan[c] = {"k": k, "d": d, "D_tr": d_tr, "D_te": d_te}

    keep1 = np.random.RandomState(seed + 2).rand(len(s1)) < k_s1
    u_pool = np.random.RandomState(seed + 3).rand(len(pool))
    keep_p = np.where(owned, keep1[np.maximum(owner, 0)], u_pool < k_pool)
    drop1 = keep1 & (np.random.RandomState(seed + 1).rand(len(s1)) < d_s1)
    final1 = keep1 & ~drop1
    own_final = owned & final1[np.maximum(owner, 0)]
    for c, pl in plan.items():
        m1, mp = c1 == c, cp == c
        n_s1, n_pool = int(final1[m1].sum()), int(keep_p[mp].sum())
        dens = n_pool / max(n_s1, 1)
        distr = 1.0 - own_final[mp & keep_p].sum() / max(n_pool, 1)
        log(f"[train] sampling {c}: k={pl['k']:.3f} d={pl['d']:.3f} kept S1={int(keep1[m1].sum())} "
            f"dropped={int(drop1[m1].sum())} final S1={n_s1} pool={n_pool} density={dens:.2f} "
            f"(train {pl['D_tr']:.2f}, test {pl['D_te'] if pl['D_te'] is None else round(pl['D_te'], 2)})"
            f" distractor share={distr:.3f}")
        pl.update(final_s1=n_s1, pool=n_pool, density=dens, distractor_share=float(distr))
        # a density gap means wrong test counts: stop before blocking. Small samples and countries
        # denser than test (d = 0 cannot fix those) only warn.
        if pl["D_te"] is not None and pl["d"] > 0:
            gap = abs(dens / pl["D_te"] - 1.0)
            if gap > 0.02:
                msg = f"[train] {c}: density {dens:.3f} is {gap:.1%} off test's {pl['D_te']:.3f}"
                if n_s1 >= 20000:
                    raise AssertionError(msg)
                log("  WARNING " + msg + " (small sample, not enforced)")
    srcs = [s1[final1].reset_index(drop=True)]
    n2 = len(s2)
    srcs.append(s2[keep_p[:n2]].reset_index(drop=True))
    srcs.append(s3[keep_p[n2:]].reset_index(drop=True))
    return srcs, plan


def legacy_sample(srcs, gt, cfg, log=print):
    """v1/v2 behaviour: global keep fraction, then drop train_drop_s1 of S1s."""
    frac = cfg["train_keep_s1"]
    if frac < 1:
        rng = np.random.RandomState(cfg["seed"] + 2)
        keep1 = rng.rand(len(srcs[0])) < frac
        kept = set(srcs[0].entity_id.values[keep1])
        srcs[0] = srcs[0][keep1].reset_index(drop=True)
        owned = {b for a, ms in zip(gt.source1_entity_id.values, gt.matched_entity_ids.values)
                 if ms and a in kept for b in ms.split(",")}
        for k in (1, 2):
            m = srcs[k].entity_id.isin(owned).values | (rng.rand(len(srcs[k])) < frac)
            srcs[k] = srcs[k][m].reset_index(drop=True)
        log(f"[train] train_keep_s1={frac}: kept {len(srcs[0])} S1, "
            f"{len(srcs[1]) + len(srcs[2])} pool records")
    if cfg["train_drop_s1"] > 0:
        keep = np.random.RandomState(cfg["seed"] + 1).rand(len(srcs[0])) >= cfg["train_drop_s1"]
        srcs[0] = srcs[0][keep].reset_index(drop=True)
        log(f"[train] dropped {int((~keep).sum())} S1 records; their matches become distractors")
    return srcs


def prepare_split(split, data_dir, work, cfg, n_jobs, log=print):
    t0 = time.time()
    d = os.path.join(data_dir, split)
    srcs = [read_source(os.path.join(d, f"{split}_source{k}.tsv")) for k in (1, 2, 3)]
    gt = None
    extra_meta = {}
    if split == "train":
        gt = read_ground_truth(os.path.join(d, "train_ground_truth.tsv"))
        if cfg["train_sampling"] == "match_test":
            srcs, plan = sample_like_test(srcs, gt, test_counts(data_dir), cfg, log)
            extra_meta["sampling"] = plan
        elif cfg["train_sampling"] == "legacy":
            srcs = legacy_sample(srcs, gt, cfg, log)
        else:
            raise ValueError(f"unknown train_sampling: {cfg['train_sampling']}")
    n1, n2, n3 = (len(s) for s in srcs)
    df = pd.concat(srcs, ignore_index=True)
    log(f"[{split}] read S1={n1} S2={n2} S3={n3} ({time.time() - t0:.0f}s)")

    if split == "train":
        gt = gt[gt.source1_entity_id.isin(set(srcs[0].entity_id))]
        lex = learn_lexicon(srcs[0], pd.concat(srcs[1:], ignore_index=True), gt, cfg, log)
        work.save_json("lexicon.json", lex)
    else:
        lex = work.load_json("lexicon.json")
    del srcs

    cols = parse_frame(df, lex, n_jobs)
    log(f"[{split}] parsed ({time.time() - t0:.0f}s)")
    n = len(df)
    ckeys = _ckey(df.country.values)
    country_codes, countries = pd.factorize(ckeys)
    country_codes = country_codes.astype(np.int16)
    if "state_fill" in nz.fixes_on(lex):
        # S1 records always name city and state: a city word -> state table from this split's
        # S1s fills records without a state (test inputs, no labels -- like IDF)
        table = nz.state_fill_table(cols["a_tokens"], cols["state"], ckeys, n1)
        filled = np.asarray(nz.fill_states(cols["a_tokens"], cols["state"], ckeys, table), bool)
        msg = [f"{c}: {int(filled[(country_codes == i)].sum())}" for i, c in enumerate(countries)]
        log(f"[{split}] state_fill: {len(table)} words in table; filled " + ", ".join(msg))
    arr = {}
    for out_name, key in STR_FIELDS.items():
        arr[out_name + "_p"], arr[out_name + "_b"] = bytes_csr(cols[key])
    per_country = cfg["idf_scope"] == "country"
    if cfg["idf_scope"] not in ("country", "split"):
        raise ValueError(f"unknown idf_scope: {cfg['idf_scope']}")
    grp = country_codes if per_country else None
    arr["nt_p"], arr["nt_d"], name_vocab, name_grp = token_csr(cols["tokens"], grp)
    arr["pt_p"], arr["pt_d"], _, _ = token_csr(cols["phon"])
    arr["at_p"], arr["at_d"], addr_vocab, addr_grp = token_csr(cols["a_tokens"], grp)
    arr["nu_p"], arr["nu_d"] = num_csr(cols["nums"])
    arr["nv_p"], arr["nv_b"] = bytes_csr(name_vocab)
    docs = np.bincount(country_codes, minlength=len(countries))
    for key, ids, vocab, g in (("n_idf", arr["nt_d"], name_vocab, name_grp),
                               ("a_idf", arr["at_d"], addr_vocab, addr_grp)):
        arr[key] = (idf_grouped(ids, g, docs) if g is not None
                    else idf_from_csr(None, ids, len(vocab), n))
    # stored for v4 (not used by v3's model)
    arr["nn_p"], arr["nn_d"] = num_csr(cols["n_nums"])
    arr["gw_p"], gw = num_csr(cols["gw"])
    arr["gw_d"] = gw.astype(np.int8)
    arr["ini_p"], arr["ini_b"] = bytes_csr(cols["ini"])
    arr["legal"], _ = codes_with_empty(cols["legal"])
    arr["state"], state_names = codes_with_empty(cols["state"])
    arr["country"] = country_codes
    arr["src"] = np.concatenate([np.full(n1, 1, np.int8), np.full(n2, 2, np.int8),
                                 np.full(n3, 3, np.int8)])
    for k in ("is_domain", "is_indic", "masked", "a_empty"):
        arr[k] = np.asarray(cols[k], np.int8)
    arr["n_ntok"] = np.diff(arr["nt_p"]).astype(np.int16)
    # how many S1 records share this (country, sorted core name) -- term-frequency signal
    key = pd.Series(cols["sorted"], dtype=object) + "|" + pd.Series(arr["country"]).astype(str)
    s1_counts = key.iloc[:n1].value_counts()
    arr["name_freq"] = key.map(s1_counts).fillna(0).to_numpy(np.int32)
    ids = df.entity_id.values.astype(str)
    arr["ids"] = np.array(ids, dtype=f"S{max(len(x) for x in ids)}")
    work.save_arrays(f"{split}/rec", arr)
    meta = {"n1": n1, "n2": n2, "n3": n3, "countries": list(countries),
            "states": list(state_names), **extra_meta}  # state code i -> states[i]
    work.save_json(f"{split}/meta.json", meta)

    if split == "train":
        pos = {e: i for i, e in enumerate(ids)}
        pool_true = np.full(n, -1, np.int32)
        n_true = np.zeros(n1, np.int16)
        for a, ms in zip(gt.source1_entity_id.values, gt.matched_entity_ids.values):
            ia = pos.get(a)
            if ia is None or not ms:
                continue
            for b in ms.split(","):
                ib = pos.get(b)
                if ib is not None:
                    pool_true[ib] = ia
                    n_true[ia] += 1
        work.save_arrays("train/truth", {"pool_true": pool_true, "n_true": n_true})
    log(f"[{split}] prepare done in {time.time() - t0:.0f}s; name vocab={len(name_vocab)} "
        f"addr vocab={len(addr_vocab)}; idf_scope={cfg['idf_scope']}")
