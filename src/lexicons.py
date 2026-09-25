"""Hand-written seed lexicons (general linguistic/geographic knowledge, no external lookups).

These are only seeds: `lexlearn.py` mines further substitutions from the training pairs.
Country-specific tables are keyed by the lower-cased `country` label; a country without a
table simply falls back to generic token handling, so the pipeline stays open-set.
"""

# ---------------------------------------------------------------- business names
LEGAL = {
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd",
    "pvt": "pvt", "private": "pvt", "pte": "pvt", "pvtltd": "pvt ltd",
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co",
    "llc": "llc", "pllc": "pllc", "llp": "llp", "lllp": "lllp", "lp": "lp",
    "pc": "pc", "plc": "plc", "pa": "pa",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "spa": "spa", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "sa": "sa", "ei": "ei", "eirl": "eirl", "selarl": "selarl",
    "scop": "scop", "scm": "scm", "sca": "sca",
}

NAME_DROP = {
    "the", "and", "of", "sri", "shri", "shree", "smt", "ms",
    "de", "du", "des", "la", "le", "les", "et", "cie", "l", "d", "a", "an",
}

ALIAS_PATTERN = (r"\s+(?:a\s*/\s*k\s*/\s*a|aka|f\s*/\s*k\s*/\s*a|fka|formerly known as|formerly|"
                 r"t\s*/\s*a|trading as|d\s*/\s*b\s*/\s*a|dba|doing business as)\s+")

DOMAIN_TLDS = r"(?:com|net|org|in|co\.in|co|fr|biz|info|io|us)"

# look-alike digits inside alphabetic words: KEYST0NE -> keystone
HOMOGLYPH = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "8": "b", "@": "a", "$": "s"}
# look-alike digits at a word edge (5ervices, Regiona1, 6roup): the same plus 2, 6, 7, 9
EDGE_DIGIT = {"0": "o", "1": "l", "2": "z", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t",
              "8": "b", "9": "g"}

# group words stored for v4 (kept in the core name): variant -> canonical
GROUP_WORDS = {"holding": "holding", "holdings": "holding", "group": "group", "groupe": "group",
               "international": "international", "intl": "international", "global": "global",
               "developpement": "development", "development": "development",
               "participations": "participations", "france": "france", "india": "india",
               "usa": "usa", "america": "america", "worldwide": "worldwide"}
GROUP_IDS = {w: i for i, w in enumerate(sorted(set(GROUP_WORDS.values())))}

# ---------------------------------------------------------------- addresses
ADDR_MAP = {
    # street types (US / India / generic)
    "rd": "road", "road": "road", "marg": "road",
    "str": "street", "street": "street",
    "ave": "avenue", "av": "avenue", "aven": "avenue", "avenue": "avenue",
    "dr": "drive", "drv": "drive", "drive": "drive",
    "ln": "lane", "lane": "lane",
    "ct": "court", "crt": "court", "court": "court",
    "blvd": "boulevard", "bd": "boulevard", "boul": "boulevard", "boulevard": "boulevard",
    "pl": "place", "place": "place",
    "hwy": "highway", "hiway": "highway", "highway": "highway",
    "pkwy": "parkway", "parkway": "parkway",
    "cir": "circle", "circle": "circle",
    "ter": "terrace", "terr": "terrace", "terrace": "terrace",
    "trl": "trail", "trail": "trail",
    "sq": "square", "square": "square",
    "expy": "expressway", "fwy": "freeway",
    "rte": "route", "route": "route",
    "cres": "crescent", "pt": "point", "mt": "mount", "ft": "fort",
    # directions
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    # units / buildings
    "apt": "apartment", "fl": "floor", "flr": "floor", "bldg": "building",
    "blk": "block", "rm": "room", "hno": "house", "h": "house",
    "nr": "near", "opp": "opposite", "opposite": "opposite",
    # French street types
    "imp": "impasse", "ch": "chemin", "chem": "chemin", "all": "allee",
    "q": "quai", "fbg": "faubourg", "crs": "cours", "sen": "sente",
    # Indian city aliases (old / new names)
    "bombay": "mumbai", "bangalore": "bengaluru", "calcutta": "kolkata", "madras": "chennai",
    "poona": "pune", "gurgaon": "gurugram", "baroda": "vadodara", "mysore": "mysuru",
    "trivandrum": "thiruvananthapuram", "cochin": "kochi", "trichy": "tiruchirappalli",
    "benares": "varanasi", "allahabad": "prayagraj", "pondicherry": "puducherry",
    "simla": "shimla", "vizag": "visakhapatnam", "belgaum": "belagavi", "mangalore": "mangaluru",
}

# tokens that carry no identity
ADDR_DROP = {"null", "na", "none", "nil", "no", "number", "num", "the", "of", "and",
             "de", "du", "des", "la", "le", "les", "d", "l"}
# place-type words written by one source only (US / India); "city corporation" is a phrase
PLACE_DROP = {"cdp", "county", "township", "village", "borough"}
PLACE_COUNTRIES = {"us", "india"}
# French house-number suffixes (2 bis rue ...)
FR_SUFFIX = {"bis", "ter", "quater"}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
    "pr": "puerto rico",
}

INDIA_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachal pradesh", "jh": "jharkhand", "ka": "karnataka", "kl": "kerala",
    "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya",
    "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "odisha", "pb": "punjab",
    "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "ts": "telangana", "tg": "telangana",
    "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand", "ut": "uttarakhand",
    "wb": "west bengal", "dl": "delhi", "jk": "jammu and kashmir", "la": "ladakh",
    "py": "puducherry", "ch": "chandigarh", "an": "andaman and nicobar islands",
    "dn": "dadra and nagar haveli and daman and diu", "dd": "dadra and nagar haveli and daman and diu",
    "ld": "lakshadweep",
}
INDIA_STATE_VARIANTS = {"orissa": "odisha", "uttaranchal": "uttarakhand", "pondicherry": "puducherry",
                        "new delhi": "delhi", "nct of delhi": "delhi", "jammu kashmir": "jammu and kashmir"}

# French regions, plus departments mapped to their region (region and department get swapped)
FRANCE_REGIONS = [
    "auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire", "corse",
    "grand est", "hauts de france", "ile de france", "normandie", "nouvelle aquitaine",
    "occitanie", "pays de la loire", "provence alpes cote d azur",
]
FRANCE_DEPARTMENTS = {
    "nouvelle aquitaine": ["gironde", "landes", "pyrenees atlantiques", "dordogne", "lot et garonne",
                           "charente", "charente maritime", "deux sevres", "vienne", "haute vienne",
                           "creuse", "correze"],
    "hauts de france": ["nord", "pas de calais", "somme", "oise", "aisne"],
    "pays de la loire": ["loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "ile de france": ["paris", "seine saint denis", "hauts de seine", "val de marne", "yvelines",
                      "essonne", "val d oise", "seine et marne"],
    "provence alpes cote d azur": ["bouches du rhone", "var", "alpes maritimes", "vaucluse", "paca"],
    "auvergne rhone alpes": ["rhone", "isere", "loire", "haute savoie", "savoie", "ain"],
    "occitanie": ["haute garonne", "herault", "gard"],
    "grand est": ["bas rhin", "haut rhin", "moselle", "marne"],
    "bretagne": ["ille et vilaine", "finistere", "morbihan", "cotes d armor"],
    "normandie": ["seine maritime", "calvados", "manche", "eure", "orne"],
}


def build_state_table():
    """(country, component text) -> canonical state/region name."""
    table = {}
    for code, full in US_STATES.items():
        table[("us", code)] = full
        table[("us", full)] = full
    for code, full in INDIA_STATES.items():
        table[("india", code)] = full
        table[("india", full)] = full
    for var, full in INDIA_STATE_VARIANTS.items():
        table[("india", var)] = full
    for region in FRANCE_REGIONS:
        table[("france", region)] = region
    for region, depts in FRANCE_DEPARTMENTS.items():
        for d in depts:
            table[("france", d)] = region
    return table


STATE_TABLE = build_state_table()
