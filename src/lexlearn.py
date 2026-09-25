"""Mine lexicons from training ground-truth pairs (training data only).

1. Indic-script name words -> Latin words, by positional alignment when token counts agree.
2. Indic-script address components (mostly state/city names) -> the S1 component they replace.
3. Single-token substitutions (abbreviations, aliases) that recur across many true pairs.
"""
import collections
import re

import numpy as np

from . import normalize as nz

RE_LATIN = re.compile(r"[A-Za-z]")


def _norm_comp(c):
    return " ".join(nz.RE_NONALNUM.sub(" ", nz.basic(c).replace(".", " ")).split())


def _pick(counter_of_counters, min_total, min_share):
    out = {}
    for key, ctr in counter_of_counters.items():
        total = sum(ctr.values())
        if total < min_total:
            continue
        best, cnt = ctr.most_common(1)[0]
        if cnt / total >= min_share and best:
            out[key] = best
    return out


def _sizes(scoped):
    return {sc: len(m) for sc, m in scoped.items()}


def learn_lexicon(s1, pool, gt, cfg, log=print):
    rng = np.random.RandomState(cfg["seed"])
    pairs = [(a, b) for a, ms in zip(gt.source1_entity_id.values, gt.matched_entity_ids.values)
             if ms for b in ms.split(",")]
    if len(pairs) > cfg["lex_pairs"]:
        pairs = [pairs[i] for i in rng.choice(len(pairs), cfg["lex_pairs"], replace=False)]
    s1_rows = dict(zip(s1.entity_id.values, zip(s1.business_name.values, s1.business_address.values,
                                                  s1.country.values)))
    need = {b for _, b in pairs}
    sub = pool[pool.entity_id.isin(need)]
    pool_rows = dict(zip(sub.entity_id.values, zip(sub.business_name.values,
                                                     sub.business_address.values, sub.country.values)))
    pairs = [(s1_rows[a], pool_rows[b]) for a, b in pairs if a in s1_rows and b in pool_rows]
    nz.set_lexicon({})

    # 1 + 2: Indic dictionaries
    name_ctr = collections.defaultdict(collections.Counter)
    addr_ctr = collections.defaultdict(collections.Counter)
    for (an, aa, _), (bn, ba, _) in pairs:
        if nz.RE_INDIC.search(bn):
            a_toks = nz.RE_NONALNUM.sub(" ", nz.basic(an).replace(".", "")).split()
            b_words = bn.split()
            if len(a_toks) == len(b_words):
                for w, t in zip(b_words, a_toks):
                    if nz.RE_INDIC.search(w):
                        name_ctr[w][t] += 1
        if nz.RE_INDIC.search(ba):
            b_comps = [c.strip() for c in ba.split(",")]
            b_latin = {_norm_comp(c) for c in b_comps if not nz.RE_INDIC.search(c)}
            cands = {_norm_comp(c) for c in aa.split(",")} - b_latin - {""}
            for c in b_comps:
                if nz.RE_INDIC.search(c) and not RE_LATIN.search(c):
                    for cand in cands:
                        addr_ctr[c][cand] += 1
    lex = {"name_indic": _pick(name_ctr, 3, 0.5), "addr_indic": _pick(addr_ctr, 3, 0.3),
           "name_map": {}, "addr_map": {}, "opts": nz.text_opts(cfg.get("text_off", ""))}
    nz.set_lexicon(lex)
    # lex_country: swaps are mined and applied per country, and an address swap also needs the
    # same first letter (v1 learned saint->street, city->unit on US/India and applied them to France)
    per_country = "lex_country" in nz.fixes_on(lex)

    # 3: substitution mining on parsed pairs
    min_count = cfg["lex_min_count"]
    nsub, ntot = collections.Counter(), collections.Counter()
    asub, atot = collections.Counter(), collections.Counter()
    for (an, aa, ac), (bn, ba, bc) in pairs:
        scope = ac.strip().lower() if per_country else "*"
        na, nb = nz.parse_name(an, ac), nz.parse_name(bn, bc)
        da, db = set(na["tokens"]) - set(nb["tokens"]), set(nb["tokens"]) - set(na["tokens"])
        if len(da) == 1 and len(db) == 1:
            x, y = db.pop(), da.pop()
            ntot[(scope, x)] += 1
            if x.isalpha() and y.isalpha() and x[0] == y[0]:
                nsub[(scope, x, y)] += 1
        pa, pb = nz.parse_addr(aa, ac), nz.parse_addr(ba, bc)
        da, db = set(pa["tokens"]) - set(pb["tokens"]), set(pb["tokens"]) - set(pa["tokens"])
        if len(da) == 1 and len(db) == 1:
            x, y = db.pop(), da.pop()
            atot[(scope, x)] += 1
            if x.isalpha() and y.isalpha() and (x[0] == y[0] or not per_country):
                asub[(scope, x, y)] += 1
    for (scope, x, y), c in nsub.items():
        if c >= min_count and c / ntot[(scope, x)] >= 0.5:
            lex["name_map"].setdefault(scope, {})[x] = y
    for (scope, x, y), c in asub.items():
        if c >= 2 * min_count and c / atot[(scope, x)] >= 0.5:
            lex["addr_map"].setdefault(scope, {})[x] = y
    # no chains: a target must not itself be rewritten
    for k in ("name_map", "addr_map"):
        lex[k] = {sc: {x: y for x, y in m.items() if y not in m} for sc, m in lex[k].items()}
    log(f"lexicon: {len(lex['name_indic'])} indic name words, {len(lex['addr_indic'])} indic "
        f"address components, name subs {_sizes(lex['name_map'])}, address subs "
        f"{_sizes(lex['addr_map'])} (from {len(pairs)} pairs); text fixes off: {lex['opts']['off']}")
    for k in ("name_map", "addr_map"):
        for sc, m in lex[k].items():
            log(f"  {k}[{sc}]: " + ", ".join(f"{x}->{y}" for x, y in sorted(m.items())[:60]))
    return lex
