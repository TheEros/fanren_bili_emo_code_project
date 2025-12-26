
import argparse
import json
import hashlib
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


from src.emo_model import load_ollama, predict_posnegneu

from src.io import read_manifest, read_csv_auto
from src.norm import normalize_text, is_gibberish_or_empty, norm_for_burst
from src.features import extract_features
from src.emo_lex import load_emo_lexicon, build_emo_index, predict_emotion_lex, combine_emo
from src.func_tag import tag_danmu_func, tag_comment_func
from src.analysis import (
    danmaku_basic_stats,
    comment_basic_stats,
    dist_table,
    curve_minute,
    detect_burst_2s,
    top_terms,
)

DANMU_KEEP = ["id","progress","mode","fontsize","color","midHash","content","date","time"]
COMMENT_KEEP = ["rpid","parent","content","like","ctime","fans_grade","level","mid"]

def ensure_dirs(outdir: Path):
    for sub in ["clean","labeled","tables","figs"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)

def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def write_run_artifacts(outdir: Path, args: argparse.Namespace):
    """写出可复现元信息（不影响现有 tables 读取逻辑，只新增文件）。"""
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(getattr(args, "manifest", "manifest.csv"))
    lexicon_path = Path(getattr(args, "lexicon", "emo_lexicon.csv"))

    # 快照文件（用于论文复现）
    if manifest_path.exists():
        shutil.copyfile(manifest_path, outdir / "manifest_snapshot.csv")
    if lexicon_path.exists():
        shutil.copyfile(lexicon_path, outdir / "lexicon_snapshot.csv")

    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "args": args_dict,
        "manifest": {
            "path": str(manifest_path),
            "md5": file_md5(manifest_path) if manifest_path.exists() else None,
        },
        "lexicon": {
            "path": str(lexicon_path),
            "md5": file_md5(lexicon_path) if lexicon_path.exists() else None,
        },
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def _add_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    feats = df[text_col].map(extract_features).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

def clean_danmaku(df: pd.DataFrame, episode_id: str) -> pd.DataFrame:
    missing = [c for c in DANMU_KEEP if c not in df.columns]
    if missing:
        raise ValueError(f"Danmaku missing columns: {missing}")

    out = df[DANMU_KEEP].copy()
    out["episode_id"] = str(episode_id)
    out["content"] = out["content"].map(normalize_text)
    out = out[~out["content"].map(is_gibberish_or_empty)].copy()

    out["sec"] = pd.to_numeric(out["progress"], errors="coerce") / 1000.0
    out["minute"] = (out["sec"] // 60).fillna(-1).astype(int)
    out["sec_bin"] = (out["sec"] // 2).fillna(-1).astype(int)
    out["norm_content"] = out["content"].map(norm_for_burst)

    out = _add_features(out, "content")
    return out

def clean_comment(df: pd.DataFrame, episode_id: str) -> pd.DataFrame:
    missing = [c for c in COMMENT_KEEP if c not in df.columns]
    if missing:
        raise ValueError(f"Comment missing columns: {missing}")

    out = df[COMMENT_KEEP].copy()
    out["episode_id"] = str(episode_id)
    out["content"] = out["content"].map(normalize_text)
    out = out[~out["content"].map(is_gibberish_or_empty)].copy()

    out["ctime_dt"] = pd.to_datetime(pd.to_numeric(out["ctime"], errors="coerce"), unit="s", errors="coerce")
    out["parent"] = pd.to_numeric(out["parent"], errors="coerce").fillna(0).astype(int)
    out["like"] = pd.to_numeric(out["like"], errors="coerce").fillna(0).astype(int)

    out["is_root"] = (out["parent"] == 0).astype(int)
    out["is_reply"] = (out["parent"] != 0).astype(int)

    reply_cnt = out[out["parent"] != 0].groupby("parent")["rpid"].count().rename("reply_count")
    out = out.merge(reply_cnt, left_on="rpid", right_index=True, how="left")
    out["reply_count"] = out["reply_count"].fillna(0).astype(int)

    out = _add_features(out, "content")
    return out

def label_danmaku(
    d: pd.DataFrame,
    emo_index: dict,
    ollama_client=None,
    ollama_workers: int = 8,
    ollama_only_for_other: bool = True,
    ollama_progress: bool = True,
) -> pd.DataFrame:
    d = d.copy()
    d["lex_emo"] = d["content"].map(lambda x: predict_emotion_lex(x, emo_index))
    d["model_used"] = False
    d["model_emo"] = "neu"  # 默认不启用模型情绪

    if ollama_client is not None:
        if ollama_only_for_other:
            mask = (d["lex_emo"] == "other")
            if mask.any():
                d.loc[mask, "model_emo"] = predict_posnegneu(
                    d.loc[mask, "content"].tolist(),
                    ollama_client,
                    workers=ollama_workers,
                    progress=ollama_progress,
                    progress_prefix="Ollama(danmu)",
                )
                d.loc[mask, "model_used"] = True
        else:
            d["model_emo"] = predict_posnegneu(
                d["content"].tolist(),
                ollama_client,
                workers=ollama_workers,
                progress=ollama_progress,
                progress_prefix="Ollama(danmu)",
            )
            d["model_used"] = True

    d["emo"] = d.apply(lambda r: combine_emo(r["lex_emo"], r["model_emo"]), axis=1)
    d["func"] = d.apply(lambda r: tag_danmu_func(r["content"], r["emo"]), axis=1)
    return d

def label_comment(
    c: pd.DataFrame,
    emo_index: dict,
    ollama_client=None,
    ollama_workers: int = 8,
    ollama_only_for_other: bool = True,
    ollama_progress: bool = True,
) -> pd.DataFrame:
    c = c.copy()
    c["lex_emo"] = c["content"].map(lambda x: predict_emotion_lex(x, emo_index))
    c["model_used"] = False
    c["model_emo"] = "neu"  # 默认不启用模型情绪

    if ollama_client is not None:
        if ollama_only_for_other:
            mask = (c["lex_emo"] == "other")
            if mask.any():
                c.loc[mask, "model_emo"] = predict_posnegneu(
                    c.loc[mask, "content"].tolist(),
                    ollama_client,
                    workers=ollama_workers,
                    progress=ollama_progress,
                    progress_prefix="Ollama(comment)",
                )
                c.loc[mask, "model_used"] = True
        else:
            c["model_emo"] = predict_posnegneu(
                c["content"].tolist(),
                ollama_client,
                workers=ollama_workers,
                progress=ollama_progress,
                progress_prefix="Ollama(comment)",
            )
            c["model_used"] = True

    c["emo"] = c.apply(lambda r: combine_emo(r["lex_emo"], r["model_emo"]), axis=1)
    c["func"] = c.apply(lambda r: tag_comment_func(r["content"], r["emo"]), axis=1)
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--lexicon", default="emo_lexicon.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--topk_terms", type=int, default=80)
    ap.add_argument("--use_ollama", action="store_true",
                help="使用本地 Ollama 生成 model_emo（pos/neg/neu），并与 lex_emo combine_emo 融合")
    ap.add_argument("--ollama_model", default="qwen3:8b",
                help="Ollama 模型名，例如 qwen3:8b / qwen2.5:7b-instruct / llama3.1:8b-instruct")
    ap.add_argument("--ollama_base_url", default="http://localhost:11434",
                help="Ollama 服务地址，默认 http://localhost:11434")
    ap.add_argument("--ollama_workers", type=int, default=8,
                help="并发线程数（建议 4~16，视机器与模型而定）")
    ap.add_argument("--ollama_scope", choices=["other", "all"], default="other",
                help="调用范围：other=仅对 lex_emo=other 的文本调用（默认/推荐）；all=对所有文本调用（很慢）")
    ap.add_argument("--ollama_no_progress", action="store_true",
                help="关闭 Ollama 预测进度条（默认开启）")

    args = ap.parse_args()
    ollama_client = None
    if getattr(args, "use_ollama", False):
        ollama_client = load_ollama(
            model=getattr(args, "ollama_model", "qwen2.5:7b-instruct"),
            base_url=getattr(args, "ollama_base_url", "http://localhost:11434"),
        )
    outdir = Path(args.outdir)
    ensure_dirs(outdir)

    # P0：可复现元信息（快照 + run_config.json）
    # 说明：写在 outdir 根目录，不影响现有 tables/figs 的兼容性。
    write_run_artifacts(outdir, args)

    mf = read_manifest(args.manifest)
    lex_df = load_emo_lexicon(args.lexicon)
    emo_index = build_emo_index(lex_df)

    episode_stats_rows = []

    for _, row in mf.iterrows():
        ep = str(row["episode_id"])
        title = str(row.get("title", ""))
        plot_func = str(row.get("plot_func", ""))

        d_raw = read_csv_auto(row["danmaku_path"])
        c_raw = read_csv_auto(row["comment_path"])

        d = clean_danmaku(d_raw, ep)
        c = clean_comment(c_raw, ep)

        # P0：清洗报告（raw/kept/dropped）
        cleaning_rows = [
            {
                "episode_id": ep,
                "dataset": "danmaku",
                "raw_n": int(len(d_raw)),
                "kept_n": int(len(d)),
                "dropped_n": int(max(0, len(d_raw) - len(d))),
                "kept_ratio": float(len(d) / max(1, len(d_raw))),
            },
            {
                "episode_id": ep,
                "dataset": "comment_all",
                "raw_n": int(len(c_raw)),
                "kept_n": int(len(c)),
                "dropped_n": int(max(0, len(c_raw) - len(c))),
                "kept_ratio": float(len(c) / max(1, len(c_raw))),
            },
        ]
        pd.DataFrame(cleaning_rows).to_csv(outdir / "tables" / f"cleaning_report_ep{ep}.csv", index=False)

        ds = danmaku_basic_stats(d)
        cs = comment_basic_stats(c)
        episode_stats_rows.append({
            "episode_id": ep,
            "title": title,
            "plot_func": plot_func,
            "danmu_total": ds["danmu_total"],
            "minute_avg_density": float(ds["minute_avg_density"]),
            "root_cnt": cs["root_cnt"],
            "reply_cnt": cs["reply_cnt"],
        })

        # 步骤2-2：弹幕基础统计
        with open(outdir / "tables" / f"ep{ep}_danmaku_basic_stats.json", "w", encoding="utf-8") as f:
            json.dump(ds, f, ensure_ascii=False, indent=2)

        # 3.2.1：高频词粗看
        top_terms(d["content"].tolist(), topk=args.topk_terms).to_csv(
            outdir / "tables" / f"ep{ep}_top_terms_danmaku.csv",
            index=False,
            encoding="utf-8-sig",
        )
        top_terms(c["content"].tolist(), topk=args.topk_terms).to_csv(
            outdir / "tables" / f"ep{ep}_top_terms_comment.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 标签
        d_lab = label_danmaku(d, emo_index, ollama_client=ollama_client, ollama_workers=args.ollama_workers, ollama_only_for_other=(args.ollama_scope == 'other'), ollama_progress=(not args.ollama_no_progress))
        c_lab = label_comment(c, emo_index, ollama_client=ollama_client, ollama_workers=args.ollama_workers, ollama_only_for_other=(args.ollama_scope == 'other'), ollama_progress=(not args.ollama_no_progress))

        # 存清洗/标注
        d.to_csv(outdir / "clean" / f"danmaku_ep{ep}_clean.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
        c.to_csv(outdir / "clean" / f"comment_ep{ep}_clean.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
        d_lab.to_csv(outdir / "labeled" / f"danmaku_ep{ep}_labeled.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
        c_lab.to_csv(outdir / "labeled" / f"comment_ep{ep}_labeled.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")

        # top roots
        cs["top_like_roots"].to_csv(outdir / "tables" / f"ep{ep}_top_like_roots.csv", index=False, encoding="utf-8-sig")
        cs["top_reply_roots"].to_csv(outdir / "tables" / f"ep{ep}_top_reply_roots.csv", index=False, encoding="utf-8-sig")

        # 分布与曲线与刷屏
        dist_table(d_lab, "emo").to_csv(outdir / "tables" / f"danmaku_emo_dist_ep{ep}.csv", index=False, encoding="utf-8-sig")
        dist_table(d_lab, "func").to_csv(outdir / "tables" / f"danmaku_func_dist_ep{ep}.csv", index=False, encoding="utf-8-sig")
        root = c_lab[c_lab["parent"] == 0]
        reply = c_lab[c_lab["parent"] != 0]

        # 评论：根评/回复 分布（P0新增：支持论文“根评 vs 回复”对照）
        dist_table(root, "emo").to_csv(outdir / "tables" / f"comment_root_emo_dist_ep{ep}.csv", index=False, encoding="utf-8-sig")
        dist_table(reply, "emo").to_csv(outdir / "tables" / f"comment_reply_emo_dist_ep{ep}.csv", index=False, encoding="utf-8-sig")
        dist_table(root, "func").to_csv(outdir / "tables" / f"comment_root_func_dist_ep{ep}.csv", index=False, encoding="utf-8-sig")
        dist_table(reply, "func").to_csv(outdir / "tables" / f"comment_reply_func_dist_ep{ep}.csv", index=False, encoding="utf-8-sig")

        # --- Ollama 极性（model_emo）与覆盖率（model_used） ---
        if "model_emo" in d_lab.columns:
            dist_table(d_lab[d_lab.get("model_used", False) == True], "model_emo").to_csv(
                outdir / "tables" / f"danmaku_model_emo_dist_ep{ep}.csv", index=False, encoding="utf-8-sig"
            )
        if "model_emo" in c_lab.columns:
            dist_table(c_lab[(c_lab["parent"] == 0) & (c_lab.get("model_used", False) == True)], "model_emo").to_csv(
                outdir / "tables" / f"comment_root_model_emo_dist_ep{ep}.csv", index=False, encoding="utf-8-sig"
            )
            dist_table(c_lab[(c_lab["parent"] != 0) & (c_lab.get("model_used", False) == True)], "model_emo").to_csv(
                outdir / "tables" / f"comment_reply_model_emo_dist_ep{ep}.csv", index=False, encoding="utf-8-sig"
            )

        danmu_total = int(len(d_lab))
        danmu_used = int(d_lab.get("model_used", False).sum()) if "model_used" in d_lab.columns else 0
        root_total = int(len(root))
        root_used = int(root.get("model_used", False).sum()) if "model_used" in root.columns else 0
        reply_total = int(len(reply))
        reply_used = int(reply.get("model_used", False).sum()) if "model_used" in reply.columns else 0

        usage = pd.DataFrame([
            {"dataset": "danmaku", "total": danmu_total, "model_used": danmu_used, "ratio": (danmu_used / danmu_total) if danmu_total else 0.0},
            {"dataset": "comment_root", "total": root_total, "model_used": root_used, "ratio": (root_used / root_total) if root_total else 0.0},
            {"dataset": "comment_reply", "total": reply_total, "model_used": reply_used, "ratio": (reply_used / reply_total) if reply_total else 0.0},
        ])
        usage.to_csv(outdir / "tables" / f"model_usage_ep{ep}.csv", index=False, encoding="utf-8-sig")

        dist_table(c_lab, "func").to_csv(outdir / "tables" / f"comment_func_dist_ep{ep}.csv", index=False, encoding="utf-8-sig")

        curve_minute(d_lab, "emo").to_csv(outdir / "tables" / f"danmaku_minute_emo_curve_ep{ep}.csv", index=False, encoding="utf-8-sig")
        curve_minute(d_lab, "func").to_csv(outdir / "tables" / f"danmaku_minute_func_curve_ep{ep}.csv", index=False, encoding="utf-8-sig")

        detect_burst_2s(d_lab, min_cnt=6).to_csv(outdir / "tables" / f"danmaku_burst_2s_ep{ep}.csv", index=False, encoding="utf-8-sig")

        # style dist json（兼容你之前文件名）
        with open(outdir / "tables" / f"ep{ep}_danmaku_style_dist.json", "w", encoding="utf-8") as f:
            json.dump({
                "mode_dist": ds.get("mode_dist", {}),
                "fontsize_dist": ds.get("fontsize_dist", {}),
                "color_top10": ds.get("color_top10", {}),
            }, f, ensure_ascii=False, indent=2)

    pd.DataFrame(episode_stats_rows).to_csv(outdir / "tables" / "episode_stats.csv", index=False, encoding="utf-8-sig")
    print("Done. Outputs in:", outdir)

if __name__ == "__main__":
    main()