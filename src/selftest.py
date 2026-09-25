"""Parser self-test: python -m src.selftest  (exit code 1 on any failure; the bootstrap stops).

1. Every v3 text fix's test case (v3 plan section 3.3).
2. v1/v2 behaviours that must not change, with the fixes on and with text_off=all.
3. text_off=all reproduces v2 on the fix cases too, and switches travel in lex["opts"].
4. Small pure-numpy checks: per-country token ids / IDF and the decision helpers.
"""
import sys

import numpy as np

from . import normalize as nz

FAILS = []


def check(label, got, want):
    if got != want:
        FAILS.append(f"{label}: got {got!r}, want {want!r}")


def name(raw, country="us"):
    return nz.parse_name(raw, country)


def addr(raw, country):
    p = nz.parse_addr(raw, country)
    return set(p["tokens"]), set(p["nums"]), p["state"]


# ---- v1/v2 behaviours that must not change (captured from v2's parser, empty lexicon)
NAME_KEEP = [
    ("Studio 54 Inc", "studio 54", "inc"),
    ("KEYST0NE Solutions LLC", "keystone solutions", "llc"),
    ("S.B.I. Life Insurance Co. Ltd.", "sbi life insurance", "co ltd"),
    ("www.acme-shop.com", "acmeshop", ""),
    ("#bestdeals", "bestdeals", ""),
    ("John Smith dba Smith Plumbing", "smith plumbing", ""),
    ("ABC Traders Pvt Ltd | XYZ", "abc traders", "ltd pvt"),
    ("Tata Motors (India)", "tata motors", ""),
    ("VIDYALAYA VIDYALAYA school", "vidyalaya school", ""),
    ("Société Générale SARL", "societe generale", "sarl"),
    ("श्री गणेश ट्रेडर्स", "ganesh tredars", ""),
    ("ਗੁਰੂ ਨਾਨਕ ਡੇਅਰੀ", "guru nanak deari", ""),
    ("Café de la Paix", "cafe paix", ""),
    ("The Bank of America Corporation", "bank america", "corp"),
    ("L'Oréal S.A.", "oreal", "sa"),
    ("AT&T Inc.", "at t", "inc"),
    ("Ms Priya Boutique", "priya boutique", ""),
    ("Route 66 Diner", "route 66 diner", ""),
    ("Planet 9 Studios", "planet 9 studios", ""),
]
ADDR_KEEP = [
    ("N° 8 Rue de la Paix, Paris, Île-de-France", "France", {"paix", "rue"}, {8}, "ile de france"),
    ("1234 N Main St, Ste 200, Springfield, Illinois, 62701", "US",
     {"main", "north", "springfield", "street", "suite"}, {200, 1234, 62701}, "illinois"),
    ("Near SBI ATM, 12/3 MG Rd, Bombay, Maharashtra", "India",
     {"atm", "mg", "mumbai", "near", "road", "sbi"}, {3, 12}, "maharashtra"),
    ("PO Box 77, 9 St Marys Rd, Tamil Nadu", "India", {"marys", "road", "street"}, {9}, "tamil nadu"),
    ("C/O Ram Kumar, 45 Park Street, Kolkata, WB", "India",
     {"kolkata", "kumar", "park", "ram", "street"}, {45}, "west bengal"),
    ("12 Avenue des Champs-Élysées, 75008 Paris, Ile de France", "France",
     {"avenue", "champs", "elysees", "paris"}, {12, 75008}, "ile de france"),
    ("", "US", set(), set(), ""),
    ("Ste 5, 100 Oak Ave, Austin, TX", "US", {"austin", "avenue", "oak", "suite"}, {5, 100}, "texas"),
    ("Plot 7, Sector 9, Phase2, Noida, UP", "India", {"noida", "plot", "sector"}, {2, 7, 9},
     "uttar pradesh"),
    ("St Louis Park, 500 Elm St, MN", "US", {"elm", "louis", "park", "saint", "street"}, {500},
     "minnesota"),
    ("3 Allée des Pins, Ste 4, Lyon, Rhône", "France", {"allee", "lyon", "pins", "suite"}, {3, 4},
     "auvergne rhone alpes"),
]


def keep_cases(tag):
    for raw, core, legal in NAME_KEEP:
        p = name(raw)
        check(f"[{tag}] name {raw!r} core", p["core"], core)
        check(f"[{tag}] name {raw!r} legal", p["legal"], legal)
    for raw, c, toks, nums, st in ADDR_KEEP:
        check(f"[{tag}] addr {raw!r}", addr(raw, c), (toks, nums, st))


def fix_cases():
    # lex_country: a swap learned on US pairs is not applied to France
    nz.set_lexicon({"opts": nz.text_opts(""), "addr_map": {"us": {"saint": "street"}}})
    check("lex_country France", addr("Saint-Nazaire, Pays de la Loire", "France")[0],
          {"saint", "nazaire"})
    check("lex_country US still swaps", addr("Saint Paul Rd, MN", "US")[0], {"street", "paul", "road"})
    nz.set_lexicon({"opts": nz.text_opts("")})
    # fr_saint
    check("fr_saint", addr("15 r. St-Honoré, 75001 Paris", "France")[0:2],
          ({"rue", "saint", "honore", "paris"}, {15, 75001}))
    check("fr_saint ste+number", addr("Ste 4, 3 rue X, Lyon", "France")[0], {"suite", "rue", "x", "lyon"})
    check("fr_saint sainte", addr("Ste Marie, Lyon", "France")[0], {"sainte", "marie", "lyon"})
    # fr_bis
    check("fr_bis", addr("2 BIS RUE X, NANTES", "France")[0:2], ({"rue", "x", "nantes"}, {2}))
    check("fr_bis ter", addr("12 ter rue Y, Lyon", "France")[0], {"rue", "y", "lyon"})
    check("fr_bis only France", addr("12 Ter Rd, TX", "US")[0], {"terrace", "road"})
    # edge_digits (names only: in addresses edge digits are glued numbers -- phase1, cour2)
    check("edge_digits", name("5ervices Regiona1 LLC")["core"], "services regional")
    check("edge_digits 1st", name("1st Choice")["core"], "1st choice")
    check("edge_digits legal", name("Acme 1lc")["legal"], "llc")
    check("edge_digits agr0", name("Agr0 Tech")["core"], "agro tech")
    check("edge_digits short", name("8km Foods")["core"], "8km foods")
    check("edge_digits 2eme", name("2eme Art")["core"], "2eme art")
    check("edge_digits addr untouched", addr("Phase1, Tower2, Noida", "India")[1], {1, 2})
    # name_noise_nums
    check("noise #n", name("Eye Clinic LLC #74779")["core"], "eye clinic")
    check("noise (ID: n)", name("Horizon Media Service (ID: 42438)")["core"], "horizon media service")
    check("noise - n", name("Chiropractic Frontier Health - 1072011349")["core"],
          "chiropractic frontier health")
    check("noise short number kept", name("Studio 54 Inc")["core"], "studio 54")
    check("noise whole name kept", name("#74779")["core"] != "", True)
    # ms
    check("ms", name("M/s Sharma Traders", "india")["core"], "sharma traders")
    check("ms dot", name("M/S. Gupta & Sons", "india")["core"], "gupta sons")
    # place_words
    check("place_words", addr("Harris County, 4 Elm Rd, TX", "US"),
          ({"harris", "elm", "road"}, {4}, "texas"))
    check("place_words cdp", addr("10 Oak St, Pine Ridge CDP, SD", "US")[0],
          {"oak", "street", "pine", "ridge"})
    check("place_words city corporation", addr("5 MG Road, Thane City Corporation, MH", "India")[0],
          {"mg", "road", "thane"})
    check("place_words not France", addr("3 rue du Village, Lyon", "France")[0], {"rue", "village", "lyon"})
    # india_floor
    check("india_floor", addr("1ST FLOOR, 12 MG ROAD, (OLD NO 94), Bengaluru, Karnataka", "India"),
          ({"mg", "road", "bengaluru"}, {12}, "karnataka"))
    for fl in ("IIFLOOR", "Fourth Floor", "010Th Floor", "ground floor", "IIND FLOOR", "Floor-2",
               "upper ground floor", "2nd flr", "Gr Floor", "G Floor"):
        check(f"india_floor {fl}", addr(f"{fl}, 7 X Rd, Pune", "India")[0:2], ({"x", "road", "pune"}, {7}))
    check("india_floor old no.15", addr("Old No.15, New No 8, Anna Salai, Chennai", "India")[1], {8})
    # state_fill
    toks = [["x", "nantes"]] * 25 + [["y", "lyon"]] * 25 + [["rue", "boileau", "nantes"], ["rue", "a", "lyon", "nantes"]]
    states = ["pays de la loire"] * 25 + ["auvergne rhone alpes"] * 25 + ["", ""]
    ctry = ["france"] * len(toks)
    table = nz.state_fill_table(toks, states, ctry, n1=50)
    filled = nz.fill_states(toks, states, ctry, table)
    check("state_fill", states[50], "pays de la loire")
    check("state_fill conflict", (states[51], filled[51]), ("", False))
    check("state_fill table min count", ("france", "x") in table, True)


def v2_equivalence():
    """text_off=all must give v2's output on the fix cases (v2 bugs included)."""
    nz.set_lexicon({"opts": nz.text_opts("all")})
    check("[off] fr_saint", addr("15 r. St-Honoré, 75001 Paris", "France")[0],
          {"rue", "street", "honore", "paris"})
    check("[off] fr_bis", addr("2 BIS RUE X, NANTES", "France")[0], {"bis", "rue", "x", "nantes"})
    check("[off] ter", addr("12 ter rue Y, Lyon", "France")[0], {"terrace", "rue", "y", "lyon"})
    check("[off] edge", name("5ervices Regiona1 LLC")["core"], "5ervices regiona1")
    check("[off] noise", name("Eye Clinic LLC #74779")["core"], "eye clinic 74779")
    check("[off] ms", name("M/s Sharma Traders")["core"], "m s sharma traders")
    check("[off] place", addr("Harris County, 4 Elm Rd, TX", "US")[0], {"harris", "county", "elm", "road"})
    check("[off] floor", addr("1ST FLOOR, 12 MG ROAD, Bengaluru, Karnataka", "India")[1], {1, 12})
    nz.set_lexicon({"opts": nz.text_opts("all"), "addr_map": {"saint": "street"}})  # flat = v2 lexicon
    check("[off] flat lexicon applies everywhere", addr("Saint-Nazaire, Lyon", "France")[0],
          {"street", "nazaire", "lyon"})
    nz.set_lexicon({})  # a lexicon without opts (v1/v2 cache) parses like v2
    check("[no opts] v2 parsing", name("5ervices")["core"], "5ervices")
    try:
        nz.text_opts("fr_saint,bogus")
        FAILS.append("text_opts accepted an unknown switch")
    except KeyError:
        pass
    check("text_opts subset", nz.text_opts("ms, fr_bis"), {"off": ["fr_bis", "ms"]})


def stored_fields():
    nz.set_lexicon({"opts": nz.text_opts("")})
    p = name("Studio 54 Groupe Holdings #12345")
    check("nn (name numbers)", p["nums"], [54])
    from .lexicons import GROUP_IDS
    check("gw (group words)", p["gw"], sorted([GROUP_IDS["group"], GROUP_IDS["holding"]]))
    check("ini", name("Association de Lesperance", "france")["ini"], "al")


def array_checks():
    from .encode import idf_grouped, token_csr
    lists = [["rue", "paris"], ["rue"], ["road"], ["rue", "road"]]
    grp = np.array([0, 0, 1, 1], np.int16)
    ptr, ids, vocab, g = token_csr(lists, grp)
    rue_ids = {ids[j] for r in range(4) for j in range(ptr[r], ptr[r + 1]) if vocab[ids[j]] == "rue"}
    check("token_csr: rue gets one id per country", len(rue_ids), 2)
    check("token_csr: vocab unprefixed", sorted(set(vocab)), ["paris", "road", "rue"])
    idf = idf_grouped(ids, g, np.array([2, 2]))
    rue_fr = [i for i in rue_ids if g[i] == 0][0]
    check("idf_grouped: rue in both FR records", round(float(idf[rue_fr]), 5),
          round(float(np.log(3 / 3) + 1), 5))
    ptr0, ids0, vocab0, g0 = token_csr(lists)
    check("token_csr without group", (len(vocab0), g0), (3, None))
    from .decide import _ef_best_k, lam_shift, prior_em, soft_exclusive
    q = soft_exclusive(np.array([0, 0, 1], np.int64), np.array([0.9, 0.7, 0.6]), 2)
    check("soft exclusivity", [round(float(x), 2) for x in q], [0.73, 0.19, 0.6])
    check("lam_shift", round(float(lam_shift(np.array([0.5]), 2.0)[0]), 4), 0.6667)
    check("has-match: p_has=0 -> empty", _ef_best_k(np.array([0.9, 0.2]), 0.0), 0)
    check("has-match off picks top", _ef_best_k(np.array([0.9, 0.2]), -1.0), 1)
    pi, lam = prior_em(np.full(1000, 0.3), 0.3)
    check("prior_em fixed point", round(lam, 3), 1.0)


def main():
    nz.set_lexicon({"opts": nz.text_opts("")})
    keep_cases("fixes on")
    nz.set_lexicon({"opts": nz.text_opts("all")})
    keep_cases("text_off=all")
    fix_cases()
    v2_equivalence()
    stored_fields()
    array_checks()
    if FAILS:
        print(f"SELFTEST FAILED ({len(FAILS)}):")
        for f in FAILS:
            print("  " + f)
        return 1
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
