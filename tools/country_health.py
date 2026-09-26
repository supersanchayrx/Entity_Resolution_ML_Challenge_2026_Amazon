"""Per-country health check without labels (v5.5 plan W0.4).

    python tools/country_health.py --work WORK [--profile v55] [--set key=value ...]

For each country of each split: the model's own expected macro F0.5 of the chosen lists (exact, from
the calibrated probabilities after exclusivity, assuming independent candidates), the share of
candidate pairs with 0.2 <= q <= 0.8, matches per S1 and the empty share. On the training split the
real OOF macro F0.5 is printed next to the expected one: where they agree for the training
countries, the expected value is a fair (label-free) estimate for a test country such as one absent
from training. Writes WORK/diag/country_health.tsv.
"""
import argparse
import os
import sys

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@njit(cache=True)
def _exp_f(ps, k):
    """E[F0.5] of predicting the first k of ps (sorted descending), candidates independent."""
    n = len(ps)
    if k == 0:
        v = 1.0
        for x in ps:
            v *= 1.0 - x
        return v
    pre = np.zeros(k + 1)
    pre[0] = 1.0
    for t in range(k):
        for c in range(t + 1, 0, -1):
            pre[c] = pre[c] * (1.0 - ps[t]) + pre[c - 1] * ps[t]
        pre[0] *= 1.0 - ps[t]
    m = n - k
    suf = np.zeros(m + 1)
    suf[0] = 1.0
    for t in range(k, n):
        for c in range(t - k + 1, 0, -1):
            suf[c] = suf[c] * (1.0 - ps[t]) + suf[c - 1] * ps[t]
        suf[0] *= 1.0 - ps[t]
    e = 0.0
    for A in range(1, k + 1):
        for B in range(m + 1):
            e += pre[A] * suf[B] * 1.25 * A / (k + 0.25 * (A + B))
    return e


@njit(parallel=True, cache=True)
def _per_s1(starts, order, qe, sel, out):
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        n = e - s
        ps = np.empty(n)
        k = 0
        for r in range(n):
            ps[r] = qe[order[s + r]]
            if sel[order[s + r]]:
                k += 1
        out[g] = _exp_f(ps, k)


def expected_f05(a, qe, sel, n1):
    """Per S1 expected F0.5 (S1s without candidates: 1, the empty list with no candidate)."""
    a64 = a.astype(np.int64)
    order = np.lexsort((-qe.astype(np.float64), a64)).astype(np.int64)
    ga = a64[order]
    starts = np.r_[0, np.nonzero(ga[1:] != ga[:-1])[0] + 1, len(ga)].astype(np.int64)
    vals = np.zeros(len(starts) - 1)
    _per_s1(starts, order, np.clip(qe.astype(np.float64), 0, 1), sel.astype(np.bool_), vals)
    out = np.ones(n1)
    out[ga[starts[:-1]]] = vals
    return out


def main():
    from src.config import load_config
    from src.decide import f05_from_counts
    from src.io_utils import Work
    from src.pipeline import _unseen_floor, decide_lam, main_lambda
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--profile")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.set, args.profile)
    work = Work(args.work)
    dec = work.load_json("model/decision.json")
    rows = []
    for split in ("train", "test"):
        meta = work.load_json(f"{split}/meta.json")
        n1, N = meta["n1"], meta["n1"] + meta["n2"] + meta["n3"]
        countries = meta["countries"]
        cand = work.load_arrays(f"{split}/cand", ["a", "b"] + (["y"] if split == "train" else []))
        a, b = cand["a"].astype(np.int64), cand["b"].astype(np.int64)
        s1_c = work.load_arrays(f"{split}/rec", ["country"])["country"][:n1].astype(np.int64)
        pair_c = s1_c[a]
        names = ["q", "sel"] + (["p_has"] if work.exists(f"{split}/scores", "p_has.npy") else [])
        sc = work.load_arrays(f"{split}/scores", names)
        lam = main_lambda(cfg, countries) if split == "test" else np.ones(len(countries))
        sel, qe = decide_lam(a, b, sc["q"], N, dec, pair_c, s1_c, countries, cfg, sc.get("p_has"), lam)
        if split == "test":
            sel = _unseen_floor(sel, qe, a, s1_c, meta, dec, cfg, lambda m: None, "health")
        ef = expected_f05(a, qe, sel, n1)
        real = None
        if split == "train":
            n_true = work.load_arrays("train/truth", ["n_true"])["n_true"]
            y = cand["y"].astype(bool)
            real = f05_from_counts(np.bincount(a[sel & y], minlength=n1), np.bincount(a[sel], minlength=n1),
                                   n_true)
        npred = np.bincount(a[sel], minlength=n1)
        for c, name in enumerate(countries):
            m = s1_c == c
            if not m.any():
                continue
            pm = pair_c == c
            unc = float(((sc["q"][pm] >= 0.2) & (sc["q"][pm] <= 0.8)).mean()) if pm.any() else float("nan")
            rows.append([split, name, int(m.sum()), f"{ef[m].mean():.5f}",
                         "" if real is None else f"{real[m].mean():.5f}", f"{unc:.4f}",
                         f"{npred[m].mean():.3f}", f"{(npred[m] == 0).mean():.4f}"])
    head = ["split", "country", "s1", "expected_f05", "oof_f05", "share_q_0.2_0.8", "matches_per_s1",
            "empty_share"]
    path = work.w("diag", "country_health.tsv")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(head) + "\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print("\t".join(head))
    for r in rows:
        print("\t".join(map(str, r)))
    print("wrote", path)


if __name__ == "__main__":
    main()
