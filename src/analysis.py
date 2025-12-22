import pandas as pd
import regex as re
from collections import Counter

# fallback tokenizer: Chinese chunks (2-6) + alnum chunks (2+)
RE_TOKEN = re.compile(r"[\p{Han}]{2,6}|[A-Za-z0-9]{2,}")

def _tokenize_fallback(s: str):
    if not isinstance(s, str):
        return []
    return RE_TOKEN.findall(s)

def _tokenize(s: str):
    # prefer jieba if available; fallback otherwise
    try:
        import jieba  # type: ignore
        return [w.strip() for w in jieba.lcut(s) if w.strip()]
    except Exception:
        return _tokenize_fallback(s)

def danmaku_basic_stats(d: pd.DataFrame) -> dict:
    stats = {}
    stats["danmu_total"] = int(len(d))
    stats["minute_avg_density"] = float(d.groupby("minute")["id"].count().mean()) if len(d) and "minute" in d.columns else 0.0
    stats["mode_dist"] = d["mode"].value_counts(normalize=True).to_dict() if "mode" in d.columns else {}
    stats["fontsize_dist"] = d["fontsize"].value_counts(normalize=True).to_dict() if "fontsize" in d.columns else {}
    stats["color_top10"] = d["color"].value_counts(normalize=True).head(10).to_dict() if "color" in d.columns else {}
    return stats

def comment_basic_stats(c: pd.DataFrame):
    roots = c[c["parent"] == 0]
    replies = c[c["parent"] != 0]
    top_like_roots = roots.sort_values("like", ascending=False).head(20)[["rpid", "like", "content"]]
    top_reply_roots = roots.sort_values("reply_count", ascending=False).head(20)[["rpid", "reply_count", "content"]]
    return {
        "root_cnt": int(len(roots)),
        "reply_cnt": int(len(replies)),
        "top_like_roots": top_like_roots,
        "top_reply_roots": top_reply_roots,
    }

def dist_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    tab = df.groupby(["episode_id", label_col]).size().reset_index(name="cnt")
    total = df.groupby(["episode_id"]).size().reset_index(name="total")
    tab = tab.merge(total, on=["episode_id"], how="left")
    tab["ratio"] = tab["cnt"] / tab["total"]
    return tab

def curve_minute(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(["episode_id", "minute", by])["id"].count().reset_index(name="cnt")
    piv = g.pivot_table(index=["episode_id", "minute"], columns=by, values="cnt", fill_value=0).reset_index()
    return piv

def detect_burst_2s(d: pd.DataFrame, min_cnt: int = 6) -> pd.DataFrame:
    g = d.groupby(["episode_id", "sec_bin", "norm_content"]).size().reset_index(name="cnt")
    g = g[(g["norm_content"].astype(str).str.len() > 0) & (g["cnt"] >= min_cnt)].copy()
    g = g.sort_values(["episode_id", "cnt"], ascending=[True, False])
    return g

def top_terms(texts, topk: int = 80, min_len: int = 2) -> pd.DataFrame:
    counter = Counter()
    for s in texts:
        for w in _tokenize(str(s)):
            w = w.strip()
            if len(w) >= min_len:
                counter[w] += 1
    items = counter.most_common(topk)
    return pd.DataFrame(items, columns=["term", "cnt"])
