import pandas as pd


def read_manifest(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_csv_auto(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
