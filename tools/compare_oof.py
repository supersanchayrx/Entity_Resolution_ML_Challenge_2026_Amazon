"""Paired comparison of two runs' OOF lists on the same training S1s (v5.5 plan W7.2).

    python tools/compare_oof.py --base WORK_A --new WORK_B

Both work folders must come from the same prepare (same sampling: same S1 ids in the same order).
Per S1, F0.5 is computed from each run's chosen pairs (train/scores/sel); the difference is averaged
over the S1s both runs scored out of fold (the holdout of either run is left out), overall and per
country, with its standard error sd(diff) / sqrt(n). A difference smaller than about 2 standard
errors is within the noise of which S1s happen to be in the sample.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def per_s1(work):
    from src.decide import f05_from_counts
    meta = work.load_json("train/meta.json")
    n1 = meta["n1"]
    cand = work.load_arrays("train/cand", ["a", "y"])
    a, y = cand["a"].astype(np.int64), cand["y"].astype(bool)
    sc = work.load_arrays("train/scores", ["sel"] + (["hold"] if work.exists("train/scores", "hold.npy") else []))
    sel = sc["sel"].astype(bool)
    n_true = work.load_arrays("train/truth", ["n_true"])["n_true"]
    f = f05_from_counts(np.bincount(a[sel & y], minlength=n1), np.bincount(a[sel], minlength=n1), n_true)
    hold = sc["hold"].astype(bool) if "hold" in sc else np.zeros(n1, bool)
    ids = work.load_arrays("train/rec", ["ids"])["ids"][:n1]
    s1_c = work.load_arrays("train/rec", ["country"])["country"][:n1].astype(np.int64)
    return f, hold, ids, s1_c, meta["countries"]


def main():
    from src.io_utils import Work
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    args = ap.parse_args()
    fa, ha, ida, ca, countries = per_s1(Work(args.base))
    fb, hb, idb, _, _ = per_s1(Work(args.new))
    if len(ida) != len(idb) or not np.array_equal(ida, idb):
        sys.exit("the two runs have different training S1s (different prepare/sampling): not comparable")
    m = ~(ha | hb)
    print(f"{'group':10} {'n_s1':>9} {'base':>9} {'new':>9} {'diff (pts)':>11} {'se (pts)':>9}")
    for name, mm in [("all", m)] + [(n, m & (ca == c)) for c, n in enumerate(countries)]:
        if not mm.any():
            continue
        d = fb[mm] - fa[mm]
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"{name:10} {int(mm.sum()):9d} {fa[mm].mean():9.5f} {fb[mm].mean():9.5f} "
              f"{100 * d.mean():+11.3f} {100 * se:9.3f}")


if __name__ == "__main__":
    main()
