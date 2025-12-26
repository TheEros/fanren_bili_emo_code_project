import regex as re

RE_BRACKET_EMO = re.compile(r"\[[^\[\]]{1,12}\]")       # [doge] [笑哭]
RE_REPEAT_CHAR = re.compile(r"(.)\1{2,}")               # 哈哈哈 / 啊啊啊
RE_AT = re.compile(r"@[\p{Han}\w_-]{1,20}")             # @xxx
# 同时支持英文与全角中文标点
RE_QM = re.compile(r"[\?？]")
RE_EM = re.compile(r"[!！]")

def extract_features(text: str) -> dict:
    t = text or ""
    return {
        "len": len(t),
        "has_bracket_emo": int(bool(RE_BRACKET_EMO.search(t))),
        "repeat_char": int(bool(RE_REPEAT_CHAR.search(t))),
        "at_cnt": len(RE_AT.findall(t)),
        "qm_cnt": len(RE_QM.findall(t)),
        "em_cnt": len(RE_EM.findall(t)),
    }
