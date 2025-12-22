import regex as re

RE_SPACE = re.compile(r"\s+")
RE_PUNC = re.compile(r"[\W_]+", re.UNICODE)
RE_KEEP = re.compile(r"[\p{Han}A-Za-z0-9]+")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = RE_SPACE.sub(" ", s)
    return s

def is_gibberish_or_empty(s: str) -> bool:
    s = normalize_text(s)
    if not s:
        return True
    keep = "".join(RE_KEEP.findall(s))
    return len(keep) == 0

def norm_for_burst(s: str) -> str:
    s = normalize_text(s).lower()
    s = RE_SPACE.sub("", s)
    s = RE_PUNC.sub("", s)
    return s
