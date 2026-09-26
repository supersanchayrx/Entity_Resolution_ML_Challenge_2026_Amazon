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
                            "nums", "state", "masked", "a_empty", "sib", "churn", "a_digits")}
    for n, a, c in zip(names, addrs, countries):
        pn = nz.parse_name(n, c)
        pa = nz.parse_addr(a, c)
        for k in ("core", "sorted", "concat", "alt", "legal", "tokens", "phon", "is_domain",
                  "is_indic", "gw", "ini", "sib", "churn"):
            cols[k].append(pn[k])
        cols["n_nums"].append(pn["nums"])
        cols["a_sorted"].append(pa["sorted"])
        cols["a_tokens"].append(pa["tokens"])
        cols["nums"].append(pa["nums"])
        cols["state"].append(pa["state"])
        cols["masked"].append(pa["masked"])
        cols["a_empty"].append(pa["empty"])
        cols["a_digits"].append(pa["digits"])
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


def _pool_plans(c1, cp, te_counts, cfg, log):
    """Per training country: [(pool name, k, d, D_tr, D_te, like)], the country's own pool first.

    Own pool: k = min(1, cap * m_te / |P_c|), d = max(0, 1 - D_tr / D_te) against the country's test.
    Each pools_extra entry {"name", "from": c, "like": L} adds a pool of c's otherwise unused S1s sized
    and densified like L's test pool (v5: us_fr). If the shares of one country sum above 1 they are
    scaled down together. Countries without test records: k = cap, d = train_drop_s1, no extras."""
    cap = float(cfg["train_size_cap"])
    extras = cfg.get("pools_extra") or []
    if extras == "auto":
        extras = auto_extra_pools(c1, cp, te_counts, cap, log)
    extras = list(extras)
    plans = {}
    for c in sorted(set(c1) | set(cp)):
        n1_tr, m_tr = int((c1 == c).sum()), int((cp == c).sum())
        d_tr = m_tr / max(n1_tr, 1)
        if not (c in te_counts and te_counts[c][0] > 0 and n1_tr > 0 and m_tr > 0):
            plans[c] = [(c, min(1.0, cap), float(cfg["train_drop_s1"]), d_tr, None, None)]
            continue
        rows = []
        for name, like in [(c, c)] + [(e["name"], e["like"]) for e in extras if e["from"] == c]:
            if like not in te_counts or te_counts[like][0] == 0:
                raise ValueError(f"pools_extra {name}: no test records for {like!r}")
            n1_te, m_te = te_counts[like]
            d_te = m_te / n1_te
            rows.append([name, min(1.0, cap * m_te / m_tr), max(0.0, 1.0 - d_tr / d_te), d_tr, d_te,
                         like])
        tot = sum(r[1] for r in rows)
        if tot > 1.0:
            log(f"  WARNING [train] {c}: pool shares sum to {tot:.3f} > 1; scaled down to fit")
            for r in rows:
                r[1] /= tot
        plans[c] = [tuple(r) for r in rows]
    return plans


def auto_extra_pools(c1, cp, te_counts, cap, log=print):
    """pools_extra="auto" (v5.5): one extra training pool per test country absent from training
    (countries are an open set), each shaped like that country's test pool and cut from the training
    country with the most S1s left over after its own pool. Name: <from>_<first 2 letters of like>."""
    train = sorted(set(c1) & set(cp))
    unseen = sorted(c for c, (n1, _) in te_counts.items() if n1 > 0 and c not in set(c1))
    if not unseen or not train:
        return []

    def leftover(c):
        n1_tr, m_tr = int((c1 == c).sum()), int((cp == c).sum())
        if c not in te_counts or te_counts[c][0] == 0 or m_tr == 0:
            return 0
        return (1.0 - min(1.0, cap * te_counts[c][1] / m_tr)) * n1_tr
    donor = max(train, key=leftover)
    if leftover(donor) <= 0:
        log(f"  pools_extra=auto: no training country has S1s to spare for {unseen}")
        return []
    names = set()
    out = []
    for u in unseen:
        name = f"{donor}_{u[:2]}"
        k = 2
        while name in names or name in train:
            k += 1
            name = f"{donor}_{u[:k]}"
        names.add(name)
        out.append({"name": name, "from": donor, "like": u})
    log(f"  pools_extra=auto: {out}")
    return out


def sample_like_test(srcs, gt, te_counts, cfg, log=print):
    """Shape each training pool like its test counterpart, in pool size and S2/S3 per S1.

    Per country, S1s are split into pools by one uniform draw u: the first k_1 of [0, 1) goes to the
    country's own pool, the next k_2 to each pools_extra pool (v5), the rest is unused. Each S1 brings
    all its matches; each distractor draws its own u (pool size like test). Then each kept S1 is
    dropped with its pool's probability d; its matches stay as distractors (density like test). The
    matches of unused S1s are never kept. With no extras this is exactly the v3 sampling.
    -> (srcs, plan by pool, pool name per kept record of each source)."""
    s1, s2, s3 = srcs
    seed = cfg["seed"]
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

    plans = _pool_plans(c1, cp, te_counts, cfg, log)
    names = [row[0] for rows in plans.values() for row in rows]
    if len(set(names)) != len(names):
        raise ValueError(f"pool names collide: {names}")
    u1 = np.random.RandomState(seed + 2).rand(len(s1))
    u_pool = np.random.RandomState(seed + 3).rand(len(pool))
    u_drop = np.random.RandomState(seed + 1).rand(len(s1))
    as1 = np.full(len(s1), -1, np.int64)        # pool index of each S1, -1 = unused
    ap = np.full(len(pool), -1, np.int64)       # pool index of each distractor
    d_s1 = np.zeros(len(s1))
    for c, rows in plans.items():
        m1, mp = c1 == c, (cp == c) & ~owned
        lo = 0.0
        for name, k, d, _, _, _ in rows:
            pi = names.index(name)
            s = m1 & (u1 >= lo) & (u1 < lo + k)
            as1[s], d_s1[s] = pi, d
            ap[mp & (u_pool >= lo) & (u_pool < lo + k)] = pi
            lo += k
    ap[owned] = as1[owner[owned]]               # matches follow their S1 (unused S1: not kept)
    keep1 = as1 >= 0
    drop1 = keep1 & (u_drop < d_s1)
    final1 = keep1 & ~drop1
    keep_p = ap >= 0
    own_final = owned & final1[np.maximum(owner, 0)]

    plan = {}
    for c, rows in plans.items():
        for name, k, d, d_tr, d_te, like in rows:
            pi = names.index(name)
            m1, mp = as1 == pi, ap == pi
            n_s1, n_pool = int((final1 & m1).sum()), int(mp.sum())
            dens = n_pool / max(n_s1, 1)
            distr = 1.0 - own_final[mp].sum() / max(n_pool, 1)
            tag = name if name == c else f"{name} (from {c}, like {like})"
            log(f"[train] sampling {tag}: k={k:.3f} d={d:.3f} kept S1={int(m1.sum())} "
                f"dropped={int((drop1 & m1).sum())} final S1={n_s1} pool={n_pool} "
                f"density={dens:.2f} (train {d_tr:.2f}, test {d_te if d_te is None else round(d_te, 2)})"
                f" distractor share={distr:.3f}")
            plan[name] = {"country": c, "like": like, "k": k, "d": d, "D_tr": d_tr, "D_te": d_te,
                          "final_s1": n_s1, "pool": n_pool, "density": dens,
                          "distractor_share": float(distr)}
            # a density gap means wrong test counts: stop before blocking. Small samples and pools
            # denser than test (d = 0 cannot fix those) only warn.
            if d_te is not None and d > 0:
                gap = abs(dens / d_te - 1.0)
                if gap > 0.02:
                    msg = f"[train] {name}: density {dens:.3f} is {gap:.1%} off test's {d_te:.3f}"
                    if n_s1 >= 20000:
                        raise AssertionError(msg)
                    log("  WARNING " + msg + " (small sample, not enforced)")
    names = np.array(names, dtype=object)
    n2 = len(s2)
    srcs = [s1[final1].reset_index(drop=True), s2[keep_p[:n2]].reset_index(drop=True),
            s3[keep_p[n2:]].reset_index(drop=True)]
    pools = [names[as1[final1]], names[ap[:n2][keep_p[:n2]]], names[ap[n2:][keep_p[n2:]]]]
    return srcs, plan, pools


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
    pool_names = None           # None: pool = country
    if split == "train":
        gt = read_ground_truth(os.path.join(d, "train_ground_truth.tsv"))
        if cfg["train_sampling"] == "match_test":
            srcs, plan, pools = sample_like_test(srcs, gt, test_counts(data_dir), cfg, log)
            extra_meta["sampling"] = plan
            if cfg.get("pools_extra"):
                pool_names = np.concatenate(pools)
        elif cfg["train_sampling"] == "legacy":
            if cfg.get("pools_extra"):
                raise ValueError("pools_extra needs train_sampling=match_test")
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
    # pool: blocking partition, IDF group, name-count scope (v5). Test and single-pool runs: the country
    if pool_names is None:
        pool_codes, pools = country_codes, list(countries)
        pool_country = list(countries)
    else:
        pool_codes, pools = pd.factorize(pd.Series(pool_names, dtype=object))
        pool_codes, pools = pool_codes.astype(np.int16), list(pools)
        pool_country = [extra_meta["sampling"][p]["country"] for p in pools]
        log(f"[{split}] pools: " + ", ".join(f"{p} ({c}): {int((pool_codes == i).sum())} records"
                                             for i, (p, c) in enumerate(zip(pools, pool_country))))
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
    grp = pool_codes if per_country else None
    arr["nt_p"], arr["nt_d"], name_vocab, name_grp = token_csr(cols["tokens"], grp)
    arr["pt_p"], arr["pt_d"], _, _ = token_csr(cols["phon"])
    arr["at_p"], arr["at_d"], addr_vocab, addr_grp = token_csr(cols["a_tokens"], grp)
    arr["nu_p"], arr["nu_d"] = num_csr(cols["nums"])
    arr["nv_p"], arr["nv_b"] = bytes_csr(name_vocab)
    docs = np.bincount(pool_codes, minlength=len(pools))
    for key, ids, vocab, g in (("n_idf", arr["nt_d"], name_vocab, name_grp),
                               ("a_idf", arr["at_d"], addr_vocab, addr_grp)):
        arr[key] = (idf_grouped(ids, g, docs) if g is not None
                    else idf_from_csr(None, ids, len(vocab), n))
    # stored for v4 (not used by v3's model)
    arr["nn_p"], arr["nn_d"] = num_csr(cols["n_nums"])
    arr["gw_p"], gw = num_csr(cols["gw"])
    arr["gw_d"] = gw.astype(np.int8)
    arr["ini_p"], arr["ini_b"] = bytes_csr(cols["ini"])
    # v5.5: sibling / churn word classes, address digits in order
    arr["sib_p"], sib = num_csr(cols["sib"])
    arr["sib_d"] = sib.astype(np.int8)
    arr["ch_p"], ch = num_csr(cols["churn"])
    arr["ch_d"] = ch.astype(np.int8)
    arr["adig_p"], arr["adig_b"] = bytes_csr(cols["a_digits"])
    arr["legal"], _ = codes_with_empty(cols["legal"])
    arr["state"], state_names = codes_with_empty(cols["state"])
    arr["country"] = country_codes
    arr["pool"] = pool_codes
    arr["src"] = np.concatenate([np.full(n1, 1, np.int8), np.full(n2, 2, np.int8),
                                 np.full(n3, 3, np.int8)])
    for k in ("is_domain", "is_indic", "masked", "a_empty"):
        arr[k] = np.asarray(cols[k], np.int8)
    arr["n_ntok"] = np.diff(arr["nt_p"]).astype(np.int16)
    # how many S1 records share this (pool, sorted core name) -- term-frequency signal
    key = pd.Series(cols["sorted"], dtype=object) + "|" + pd.Series(arr["pool"]).astype(str)
    s1_counts = key.iloc[:n1].value_counts()
    arr["name_freq"] = key.map(s1_counts).fillna(0).to_numpy(np.int32)
    ids = df.entity_id.values.astype(str)
    arr["ids"] = np.array(ids, dtype=f"S{max(len(x) for x in ids)}")
    work.save_arrays(f"{split}/rec", arr)
    meta = {"n1": n1, "n2": n2, "n3": n3, "countries": list(countries),
            "pools": pools, "pool_country": pool_country,
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
