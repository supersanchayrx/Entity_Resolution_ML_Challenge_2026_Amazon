"""Record normalization: Unicode cleanup, Indic transliteration, name and address parsing.

Pure Python + `re`; runs once per record (parallelised by the caller).
"""
import re
import unicodedata

from .lexicons import (ADDR_DROP, ADDR_MAP, ALIAS_PATTERN, CHURN_IDS, CHURN_WORDS, DOMAIN_TLDS,
                       EDGE_DIGIT, FR_SUFFIX, GROUP_IDS, GROUP_WORDS, SIB_IDS, SIBLING_WORDS, HOMOGLYPH, LEGAL, PLACE_COUNTRIES, PLACE_DROP,
                       NAME_DROP, STATE_TABLE)

# ------------------------------------------------------------------ Indic transliteration
# The Unicode blocks for Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada
# and Malayalam (U+0900..U+0D7F) share one ISCII-derived layout, so the offset inside a
# 0x80-sized block identifies the letter in every script and a single table covers all nine.
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
         0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
         0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
         0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
         0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
         0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "r",
         0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
           0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
           0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
           0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
           0x4C: "au", 0x57: "au", 0x62: "li", 0x63: "li"}
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}
_FINALS = {0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k", 0x4E: "t"}
_VIRAMA, _NUKTA = 0x4D, 0x3C
_NUKTA_MAP = {"ph": "f", "j": "z", "k": "q", "d": "r", "dh": "rh", "kh": "kh", "g": "g"}

RE_INDIC = re.compile("[ऀ-ൿ]")


def translit_indic(s):
    out = []
    pending_a = False  # inherent vowel of the last consonant, emitted only if followed by more
    for ch in s:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            off = cp & 0x7F
            if off in _CONS:
                if pending_a:
                    out.append("a")
                out.append(_CONS[off])
                pending_a = True
            elif off in _MATRAS:
                pending_a = False
                out.append(_MATRAS[off])
            elif off == _VIRAMA:
                pending_a = False
            elif off == _NUKTA:
                if out and out[-1] in _NUKTA_MAP:
                    out[-1] = _NUKTA_MAP[out[-1]]
            elif off in _VOWELS:
                if pending_a:
                    out.append("a")
                pending_a = False
                out.append(_VOWELS[off])
            elif off in _NASAL:
                if pending_a:
                    out.append("a")
                pending_a = False
                out.append(_NASAL[off])
            elif off in _FINALS:
                if pending_a:
                    out.append("a")
                pending_a = False
                out.append(_FINALS[off])
            elif 0x66 <= off <= 0x6F:
                pending_a = False
                out.append(chr(ord("0") + off - 0x66))
            elif off in (0x64, 0x65):
                pending_a = False
                out.append(" ")
            # other signs are ignored
        else:
            pending_a = False  # word-final schwa deletion
            out.append(ch)
    return "".join(out)


# ------------------------------------------------------------------ basic text cleanup
_SPECIAL = str.maketrans({"ß": "ss", "æ": "ae", "Æ": "ae", "ø": "o", "Ø": "o", "œ": "oe",
                          "Œ": "oe", "đ": "d", "Đ": "d", "ł": "l", "Ł": "l", "ı": "i", "ð": "d",
                          "þ": "th", "°": "o", "º": "o"})


def basic(s):
    """Transliterate Indic scripts, strip accents, drop remaining non-ASCII, lower-case."""
    if RE_INDIC.search(s):
        s = translit_indic(s)
    s = s.translate(_SPECIAL)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


_PH_SUBS = [("ph", "f"), ("sh", "s"), ("ch", "k"), ("kh", "k"), ("gh", "g"), ("th", "t"),
            ("dh", "d"), ("bh", "b"), ("ck", "k"), ("q", "k"), ("c", "k"), ("z", "s"),
            ("w", "v"), ("x", "ks"), ("y", "i")]


def phonetic(tok):
    """Consonant skeleton tuned for English/Indic-transliteration/French spelling variants."""
    for a, b in _PH_SUBS:
        tok = tok.replace(a, b)
    if not tok:
        return tok
    out = ["a" if tok[0] in "aeiou" else tok[0]]
    for ch in tok[1:]:
        if ch in "aeiouh" or ch == out[-1]:
            continue
        out.append(ch)
    return "".join(out)




# ------------------------------------------------------------------ text-fix switches
# Each v3 fix can be disabled by name (config text_off=a,b; "all" restores v2 parsing). The
# switches travel inside the lexicon dict (lex["opts"]) so worker processes and both splits agree.
FIXES = ("lex_country", "fr_saint", "fr_bis", "edge_digits", "name_noise_nums", "ms",
         "place_words", "india_floor", "state_fill")


def text_opts(text_off):
    """Config value of text_off -> the opts dict stored in the lexicon."""
    if str(text_off).strip() == "all":
        off = list(FIXES)
    else:
        off = [x.strip() for x in str(text_off or "").split(",") if x.strip()]
    unknown = set(off) - set(FIXES)
    if unknown:
        raise KeyError(f"unknown text_off switches: {sorted(unknown)}; known: {FIXES}")
    return {"off": sorted(off)}


def fixes_on(lex):
    """Enabled fixes; a lexicon without opts predates v3 and gets v2 parsing."""
    opts = lex.get("opts")
    if opts is None:
        return frozenset()
    return frozenset(FIXES) - set(opts.get("off", []))


# ------------------------------------------------------------------ learned lexicon
_LEX = {"name_indic": {}, "addr_indic": {}, "name_map": {}, "addr_map": {}}
_ON = frozenset()


def _scoped(m):
    """Learned swaps as {scope: {x: y}}, scope = country or "*"; pre-v3 lexicons are flat."""
    m = m or {}
    if all(isinstance(v, str) for v in m.values()):
        return {"*": dict(m)} if m else {}
    return {k: dict(v) for k, v in m.items()}


def set_lexicon(lex):
    global _LEX, _ON
    _LEX = {k: dict(lex.get(k, {})) for k in ("name_indic", "addr_indic")}
    _LEX["name_map"] = _scoped(lex.get("name_map"))
    _LEX["addr_map"] = _scoped(lex.get("addr_map"))
    _ON = fixes_on(lex)


def _swaps(kind, ckey):
    m = _LEX[kind]
    return m.get(ckey) or m.get("*") or {}


# ------------------------------------------------------------------ names
RE_NONALNUM = re.compile(r"[^a-z0-9]+")
RE_HOMO = re.compile(r"(?<=[a-z])([0134583@$])(?=[a-z])")
RE_ALIAS = re.compile(ALIAS_PATTERN)
RE_PAREN_NOISE = re.compile(r"\(\s*(?:france|india|usa|us)\s*\)")
RE_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?#?([a-z0-9][a-z0-9\-]{2,})\." + DOMAIN_TLDS +
                       r"$|^#([a-z0-9]{3,})$")
_STRIP_CHARS = " -_>*~=!:;,\"'<[]()/\\|+"
# v3 name fixes
RE_MS = re.compile(r"\bm\s*/\s*s\b\.?")
RE_NOISE = [re.compile(r"\(\s*id\s*:\s*[\[(]?\s*\d+\s*\)"),   # (ID: 42438)
            re.compile(r"(?<=\S)\s*#\s*\d+\s*$"),              # trailing #74779
            re.compile(r"(?<=\S)\s*-\s*\d{5,}\s*$")]           # trailing - 1072011349
RE_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th|er|ere|eme|e|re|ieme)$")


def edge_fix(w):
    """5ervices -> services, regiona1 -> regional, 1lc -> llc; ordinals and short words kept."""
    if RE_ORDINAL.match(w):
        return w
    pre = EDGE_DIGIT.get(w[0], "") if len(w) > 1 else ""
    core = w[1:] if pre else w
    suf = EDGE_DIGIT.get(core[-1], "") if len(core) > 1 else ""
    if suf:
        core = core[:-1]
    if not (pre or suf) or not core.isalpha():
        return w
    cand = pre + core + suf
    return cand if len(core) >= 3 or cand in LEGAL else w


def _name_tokens(part, name_map):
    """-> (core tokens, legal-form tokens, is_domain, all tokens as written)."""
    p = part.strip(_STRIP_CHARS)
    dm = RE_DOMAIN.match(p)
    if dm:
        stem = (dm.group(1) or dm.group(2)).replace("-", "")
        return [stem], [], True, [stem]
    p = RE_HOMO.sub(lambda m: HOMOGLYPH.get(m.group(1), m.group(1)), p)
    p = p.replace("&", " and ").replace(".", "")
    edge = "edge_digits" in _ON
    core, legal, full = [], [], []
    prev = None
    for t in RE_NONALNUM.sub(" ", p).split():
        if edge and not t.isalpha() and not t.isdigit():
            t = edge_fix(t)
        t = name_map.get(t, t)
        if t == prev:  # "VIDYALAYA VIDYALAYA", "PVT PVT"
            continue
        prev = t
        if t in LEGAL:
            legal.extend(LEGAL[t].split())
            full.append(t)
        elif t not in NAME_DROP:
            core.append(t)
            full.append(t)
    return core, legal, False, full


def _strip_name_noise(s):
    t = s
    if "ms" in _ON:
        t = RE_MS.sub(" ", t)
    if "name_noise_nums" in _ON:
        for rx in RE_NOISE:
            t = rx.sub(" ", t)
    return t if RE_NONALNUM.sub("", t) else s  # never strip a name down to nothing


def parse_name(raw, country=""):
    """-> dict of name fields."""
    is_indic = RE_INDIC.search(raw) is not None
    if is_indic and _LEX["name_indic"]:
        raw = " ".join(_LEX["name_indic"].get(w, w) for w in raw.split())
    s = basic(raw)
    if "|" in s:
        parts = [p for p in s.split("|") if p.strip()]
        s = parts[0] if parts else ""
    s = RE_PAREN_NOISE.sub(" ", s)
    s = _strip_name_noise(s)
    alt_s = ""
    m = RE_ALIAS.search(s)
    if m and s[m.end():].strip():
        s, alt_s = s[m.end():], s[:m.start()]
    name_map = _swaps("name_map", country.strip().lower())
    core, legal, is_domain, full = _name_tokens(s, name_map)
    if not core:
        core = sorted(set(legal)) or full
    alt = _name_tokens(alt_s, name_map)[0] if alt_s else []
    uniq = sorted(set(core))
    return {
        "core": " ".join(core),
        "sorted": " ".join(uniq),
        "concat": "".join(full) if full else "".join(core),
        "alt": " ".join(alt),
        "legal": " ".join(sorted(set(legal))),
        "tokens": uniq,
        "phon": sorted({phonetic(t) for t in uniq if not t.isdigit()}),
        "is_domain": int(is_domain),
        "is_indic": int(is_indic),
        # stored for v4, not used by v3's model
        "nums": sorted({int(t[-12:]) for t in uniq if t.isdigit()}),
        "gw": sorted({GROUP_IDS[GROUP_WORDS[t]] for t in uniq if t in GROUP_WORDS}),
        "ini": "".join(t[0] for t in core),
        # v5.5: sibling / churn word classes (lexicons.SIBLING_WORDS, CHURN_WORDS)
        "sib": sorted({SIB_IDS[SIBLING_WORDS[t]] for t in uniq if t in SIBLING_WORDS}),
        "churn": sorted({CHURN_IDS[CHURN_WORDS[t]] for t in uniq if t in CHURN_WORDS}),
    }


# ------------------------------------------------------------------ addresses
RE_POBOX = re.compile(r"\bp\s*\.?\s*o\s*\.?\s*box\s*#?\s*\d*")
RE_CARE_OF = re.compile(r"\b[cs]\s*/\s*o\b")
RE_NUM = re.compile(r"\d+")
# v3 address fixes
RE_CITY_CORP = re.compile(r"\bcity corporation\b")
_ORD_WORDS = ("first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|"
              "twelfth")
RE_FLOOR = re.compile(  # 1ST FLOOR, Fourth Floor, IIFLOOR, 010Th Floor, ground floor, Floor-2
    r"\b(?:(?:upper|lower)\s*)?(?:\d+\s*(?:st|nd|rd|th)?|(?:" + _ORD_WORDS + r")|ground|grd|gr|gnd|g|"
    r"basement|[ivx]{1,4}(?:st|nd|rd|th)?)\s*-?\s*(?:floor|flr)\b"
    r"|\b(?:floor|flr)\s*(?:-|:|no\b\.?)\s*(?:\d+|grd|ground|g)\b")
RE_OLDNO = re.compile(r"\(?\s*\bold\s*no\b\.?\s*[:\-]?\s*\d[0-9a-z/\-]*\s*\)?")  # (OLD NO 94)


def parse_addr(raw, country):
    """-> dict of address fields. Components are treated as an unordered set."""
    if not raw.strip():
        return {"sorted": "", "tokens": [], "nums": [], "state": "", "masked": 0, "empty": 1,
                "digits": ""}
    if RE_INDIC.search(raw) and _LEX["addr_indic"]:
        raw = ",".join(_LEX["addr_indic"].get(c.strip(), c) for c in raw.split(","))
    s = basic(raw)
    masked = int("#" in s)
    s = RE_POBOX.sub(" , ", s)
    s = RE_CARE_OF.sub(" ", s)
    ckey = country.strip().lower()
    fr = ckey == "france"
    fr_saint = fr and "fr_saint" in _ON
    fr_bis = fr and "fr_bis" in _ON
    place = ckey in PLACE_COUNTRIES and "place_words" in _ON
    if ckey == "india" and "india_floor" in _ON:
        s = RE_FLOOR.sub(" ", RE_OLDNO.sub(" ", s))
    addr_map = _swaps("addr_map", ckey)
    state = ""
    toks, seen, nums = [], set(), set()
    dseq = []           # v5.5: digit runs in order of appearance ("7-04" and "704" -> "704")
    for comp in s.split(","):
        clean = " ".join(RE_NONALNUM.sub(" ", comp.replace(".", " ")).split())
        if place and "city corporation" in clean:
            clean = " ".join(RE_CITY_CORP.sub(" ", clean).split())
        if not clean:
            continue
        st = STATE_TABLE.get((ckey, clean))
        if st is not None:
            state = state or st
            continue
        ws = clean.split()
        for i, w in enumerate(ws):
            if any(ch.isdigit() for ch in w):
                for d in RE_NUM.findall(w):
                    nums.add(int(d[-12:]))
                    dseq.append(d)
                continue
            if fr_bis and w in FR_SUFFIX:
                continue
            if place and w in PLACE_DROP:
                continue
            if w == "st":
                w = "saint" if (fr_saint or (i == 0 and len(ws) > 1)) else "street"
            elif w == "ste":
                w = "suite" if (i + 1 < len(ws) and ws[i + 1][0].isdigit()) else "sainte"
            elif w == "r" and i + 1 < len(ws) and ws[i + 1].isalpha():
                w = "rue"
            w = ADDR_MAP.get(w, w)
            w = addr_map.get(w, w)
            if w in ADDR_DROP or w in seen:
                continue
            seen.add(w)
            toks.append(w)
    return {"sorted": " ".join(sorted(seen)), "tokens": sorted(seen), "nums": sorted(nums),
            "state": state, "masked": masked, "empty": 0, "digits": "".join(dseq)[-24:]}


def parse_record(name, addr, country):
    return parse_name(name, country), parse_addr(addr, country)


# ------------------------------------------------------------------ state fill (fix state_fill)
def state_fill_table(tokens, states, countries, n1, min_count=20, min_share=0.95):
    """(country, address word) -> state, from the split's first n1 records (its S1s): the word is
    seen >= min_count times in S1 addresses that name a state, >= min_share of them in one."""
    tot, per = {}, {}
    for i in range(n1):
        st = states[i]
        if not st:
            continue
        c = countries[i]
        for w in tokens[i]:
            k = (c, w)
            tot[k] = tot.get(k, 0) + 1
            k2 = (c, w, st)
            per[k2] = per.get(k2, 0) + 1
    table = {}
    for (c, w, st), n in per.items():
        t = tot[(c, w)]
        if t >= min_count and n >= min_share * t:
            table[(c, w)] = st
    return table


def fill_states(tokens, states, countries, table):
    """Fill empty states in place from the table; conflicting hits leave the state empty.
    Returns a boolean list: which records were filled."""
    filled = [False] * len(states)
    for i, st in enumerate(states):
        if st:
            continue
        c = countries[i]
        hits = {table[(c, w)] for w in tokens[i] if (c, w) in table}
        if len(hits) == 1:
            states[i] = hits.pop()
            filled[i] = True
    return filled
