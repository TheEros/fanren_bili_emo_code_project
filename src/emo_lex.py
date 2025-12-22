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
    pr = priority or DEFAULT_PRIORITY
    pr_rank = {emo: i for i, emo in enumerate(pr)}
    best_emo = None
    best_rank = len(pr) + 1
    for w, emo in emo_index.items():
        if w and w in t:
            rank = pr_rank.get(emo, len(pr) + 1)
            if rank < best_rank:
                best_rank = rank
                best_emo = emo
                if best_rank == 0:
                    return best_emo
    return best_emo or "other"

def combine_emo(lex_emo: str, model_emo: str = "neu") -> str:
    if lex_emo != "other":
        return lex_emo
    if model_emo == "pos":
        return "praise"
    if model_emo == "neg":
        return "neg"
    return "other"
