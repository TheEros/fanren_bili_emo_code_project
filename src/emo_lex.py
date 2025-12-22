import pandas as pd
from typing import Dict, List

DEFAULT_PRIORITY: List[str] = ["self_mock", "laugh", "touching", "praise", "neg"]

def load_emo_lexicon(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["word"] = df["word"].astype(str)
    df["emo"] = df["emo"].astype(str)
    return df

def build_emo_index(lex_df: pd.DataFrame) -> Dict[str, str]:
    return dict(zip(lex_df["word"], lex_df["emo"]))

def predict_emotion_lex(text: str, emo_index: Dict[str, str], priority: List[str] = None) -> str:
    t = text or ""
    hits = []
    for w, emo in emo_index.items():
        if w and w in t:
            hits.append(emo)
    if not hits:
        return "other"
    pr = priority or DEFAULT_PRIORITY
    for p in pr:
        if p in hits:
            return p
    return hits[0]

def combine_emo(lex_emo: str, model_emo: str = "neu") -> str:
    if lex_emo != "other":
        return lex_emo
    if model_emo == "pos":
        return "praise"
    if model_emo == "neg":
        return "neg"
    return "other"
