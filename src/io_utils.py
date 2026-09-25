"""TSV input/output and the work-directory store."""
import csv
import json
import os

import numpy as np
import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path):
    """Read a source TSV verbatim (no quote handling, no NA coercion)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                     quoting=csv.QUOTE_NONE, encoding="utf-8")
    return df[SOURCE_COLS]


def read_ground_truth(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8")


def write_id_lists(path, header, s1_ids, id_lists):
    """Write `source1_entity_id<TAB>comma,separated,ids` rows (empty list -> empty field)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1, ids in zip(s1_ids, id_lists):
            f.write(s1 + "\t" + ",".join(ids) + "\n")


class Work:
    """Artifacts are written to `out` and read from `out` first, then each `ins` dir.

    Locally in == out. On SageMaker Processing the previous runs' artifacts are mounted
    read-only as inputs and new artifacts go to the output dir that is uploaded to S3.
    """

    def __init__(self, out, ins=()):
        self.out = out
        self.ins = [d for d in ins if d and os.path.abspath(d) != os.path.abspath(out)]
        os.makedirs(out, exist_ok=True)

    def w(self, *parts):
        path = os.path.join(self.out, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def r(self, *parts):
        for base in [self.out] + self.ins:
            path = os.path.join(base, *parts)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(os.path.join(*parts) + f" not found in {[self.out] + self.ins}")

    def exists(self, *parts):
        try:
            self.r(*parts)
            return True
        except FileNotFoundError:
            return False

    # array groups: one directory of .npy files
    def save_arrays(self, group, arrays):
        for name, arr in arrays.items():
            np.save(self.w(group, name + ".npy"), arr)

    def load_arrays(self, group, names=None, mmap=False):
        base = self.r(group)
        names = names or [f[:-4] for f in os.listdir(base) if f.endswith(".npy")]
        return {n: np.load(os.path.join(base, n + ".npy"), mmap_mode="r" if mmap else None,
                           allow_pickle=False) for n in names}

    def save_json(self, name, obj):
        with open(self.w(name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)

    def load_json(self, name):
        with open(self.r(name), encoding="utf-8") as f:
            return json.load(f)
