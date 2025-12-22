# -*- coding: utf-8 -*-
"""src.emo_model (Ollama)

用本地 Ollama 预测情绪极性：pos / neg / neu。

你现在遇到“全是 neu”，最常见原因是：
1) 请求根本没打到 Ollama（URL/网络/端口问题） -> 我们之前吞掉异常直接返回 neu
2) Ollama 版本较旧/参数不兼容（structured outputs / think 字段不支持） -> 400/报错也被吞掉
3) 模型无视指令输出长文（尤其 Qwen3 thinking）-> 解析不到标签就 neu

本版本做了**兼容与可诊断**：
- 先尝试 structured outputs（format=JSON Schema）
- 若 Ollama 返回 400（不支持 schema），自动降级到 format="json"
- 若仍失败，再降级到不传 format
- 增加 DEBUG：设置环境变量 OLLAMA_DEBUG=1 会打印少量错误与样例响应（不会刷屏）

依赖：仅标准库（urllib），无需额外 pip 包。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import re
import time
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

LABEL_RE = re.compile(r"\b(pos|neg|neu)\b", re.IGNORECASE)

CN_MAP = {
    "正面": "pos",
    "积极": "pos",
    "正向": "pos",
    "负面": "neg",
    "消极": "neg",
    "负向": "neg",
    "中性": "neu",
    "中立": "neu",
}

LABEL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["pos", "neg", "neu"]}},
    "required": ["label"],
    "additionalProperties": False,
}


def _debug_enabled() -> bool:
    return os.getenv("OLLAMA_DEBUG", "").strip() not in ("", "0", "false", "False")


def _neutralize_mode_directives(text: str) -> str:
    # 防止文本里出现 /think 触发模型模式切换（尤其 Qwen3）
    if not text:
        return ""
    return (
        text.replace("/think", "／think")
            .replace("/no_think", "／no_think")
            .replace("/set nothink", "／set nothink")
    )


def _strip_think_blocks(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE).strip()


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    timeout_s: int = 60
    max_retries: int = 2
    options: Optional[Dict[str, Any]] = None
    think: Optional[bool] = False  # 支持 thinking 的模型：False 关闭
    format_mode: str = "auto"  # auto|schema|json|none


def load_ollama(
    model: str = "qwen3:8b",
    base_url: str = "http://localhost:11434",
    timeout_s: int = 60,
    max_retries: int = 2,
    options: Optional[Dict[str, Any]] = None,
    think: Optional[bool] = False,
    format_mode: str = "auto",
) -> OllamaClient:
    if options is None:
        options = {
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 32,  # JSON 输出给点余量
        }
    return OllamaClient(
        base_url=base_url.rstrip("/"),
        model=model,
        timeout_s=timeout_s,
        max_retries=max_retries,
        options=options,
        think=think,
        format_mode=format_mode,
    )


def _build_payload(client: OllamaClient, safe_text: str, fmt: Optional[Any]) -> Dict[str, Any]:
    system = (
        "你是情绪极性分类器。"
        "你的任务是输出情绪极性 label。"
        "输出必须是 JSON 且仅包含字段 label，取值 pos/neg/neu。"
        "/no_think"
    )
    payload: Dict[str, Any] = {
        "model": client.model,
        "stream": False,
        "options": client.options or {},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f'请判断文本情绪极性并按JSON输出。文本："""{safe_text}"""'},
        ],
    }
    # think 字段在新版本 Ollama 支持；老版本可能不识别（会 400），我们在调用层自动降级
    if client.think is not None:
        payload["think"] = client.think
    if fmt is not None:
        payload["format"] = fmt
    return payload


def _call_ollama_chat(client: OllamaClient, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{client.base_url}/api/chat"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")

    last_err: Optional[Exception] = None
    for attempt in range(client.max_retries + 1):
        try:
            with urlrequest.urlopen(req, timeout=client.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            return json.loads(raw)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"Ollama call failed: {last_err}")


def _try_chat_with_fallbacks(client: OllamaClient, text: str) -> str:
    safe_text = _neutralize_mode_directives(text)

    # 决定 format 策略
    mode = (client.format_mode or "auto").lower()
    fmt_candidates: List[Tuple[str, Optional[Any]]] = []
    if mode == "schema":
        fmt_candidates = [("schema", LABEL_SCHEMA)]
    elif mode == "json":
        fmt_candidates = [("json", "json")]
    elif mode == "none":
        fmt_candidates = [("none", None)]
    else:
        # auto：schema -> json -> none
        fmt_candidates = [("schema", LABEL_SCHEMA), ("json", "json"), ("none", None)]

    last_http_400: Optional[str] = None

    for fmt_name, fmt in fmt_candidates:
        payload = _build_payload(client, safe_text, fmt)
        try:
            obj = _call_ollama_chat(client, payload)
            msg = obj.get("message") or {}
            content = str(msg.get("content", "")).strip()
            if _debug_enabled():
                print(f"[OLLAMA_DEBUG] format={fmt_name} content_sample={content[:120]!r}")
            return content
        except RuntimeError as e:
            # 可能内部是 HTTPError 400；我们无法直接拿到 code，这里做字符串兜底
            s = str(e)
            if "HTTP Error 400" in s:
                last_http_400 = s
                if _debug_enabled():
                    print(f"[OLLAMA_DEBUG] format={fmt_name} got 400, fallback next. err={s}")
                continue
            # 其它错误直接抛出，让上层降级为 neu（并在 debug 打印）
            if _debug_enabled():
                print(f"[OLLAMA_DEBUG] format={fmt_name} failed err={s}")
            raise

    # 全部失败
    raise RuntimeError(f"Ollama format fallbacks exhausted. last400={last_http_400}")


def _parse_label(content: str) -> str:
    if not content:
        return "neu"
    content = _strip_think_blocks(content)

    # JSON 优先
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            lab = str(obj.get("label", "")).strip().lower()
            if lab in ("pos", "neg", "neu"):
                return lab
    except Exception:
        pass

    # 兜底：直接找 pos/neg/neu
    m = LABEL_RE.search(content.lower())
    if m:
        lab = m.group(1).lower()
        if lab in ("pos", "neg", "neu"):
            return lab

    # 兜底：中文关键词
    for k, v in CN_MAP.items():
        if k in content:
            return v

    return "neu"



import sys

def _progress_line(done: int, total: int, prefix: str = "Ollama", extra: str = "") -> str:
    pct = (done / total * 100.0) if total else 100.0
    bar_w = 28
    filled = int(bar_w * done / total) if total else bar_w
    bar = "█" * filled + "░" * (bar_w - filled)
    msg = f"{prefix} [{bar}] {done}/{total} ({pct:5.1f}%)"
    if extra:
        msg += f"  {extra}"
    return msg

def _progress_print(line: str) -> None:
    sys.stdout.write("\r" + line)
    sys.stdout.flush()
def predict_posnegneu_one(text: str, client: OllamaClient) -> str:
    t = (text or "").strip()
    if not t or re.fullmatch(r"[\W_]+", t, flags=re.UNICODE):
        return "neu"

    try:
        content = _try_chat_with_fallbacks(client, t)
        return _parse_label(content)
    except Exception as e:
        if _debug_enabled():
            print(f"[OLLAMA_DEBUG] predict failed -> neu. err={e}")
        return "neu"


def predict_posnegneu(
    texts: List[str],
    client: OllamaClient,
    workers: int = 8,
    progress: bool = True,
    progress_prefix: str = "Ollama",
) -> List[str]:
    """批量预测 pos/neg/neu（并发 + 缓存 + 进度条）

    - progress 展示的是“唯一文本数”的完成度（因为会缓存去重）
    - 起始行会输出 raw/unique/cached/workers，方便估算耗时
    """
    if not texts:
        return []

    cache: Dict[str, str] = {}
    out: List[str] = ["neu"] * len(texts)
    todo: List[Tuple[int, str]] = []

    for i, t in enumerate(texts):
        tt = (t or "").strip()
        if tt in cache:
            out[i] = cache[tt]
        else:
            todo.append((i, tt))

    raw_total = len(texts)
    unique_total = len(todo)
    cached = raw_total - unique_total

    if unique_total == 0:
        return out

    w = max(1, int(workers or 1))

    if progress:
        extra = f"raw={raw_total} unique={unique_total} cached={cached} workers={w}"
        _progress_print(_progress_line(0, unique_total, prefix=progress_prefix, extra=extra))

    def _tick(done: int) -> None:
        if not progress:
            return
        if done == unique_total or done % 20 == 0:
            _progress_print(_progress_line(done, unique_total, prefix=progress_prefix))

    if w == 1:
        done = 0
        for i, t in todo:
            lab = predict_posnegneu_one(t, client)
            cache[t] = lab
            out[i] = lab
            done += 1
            _tick(done)
        if progress:
            sys.stdout.write("\n")
        return out

    with ThreadPoolExecutor(max_workers=w) as ex:
        futs = {ex.submit(predict_posnegneu_one, t, client): (i, t) for i, t in todo}
        done = 0
        for fut in as_completed(futs):
            i, t = futs[fut]
            try:
                lab = fut.result()
            except Exception:
                lab = "neu"
            cache[t] = lab
            out[i] = lab
            done += 1
            _tick(done)

    if progress:
        sys.stdout.write("\n")
    return out

    w = max(1, int(workers or 1))
    if w == 1:
        for i, t in todo:
            lab = predict_posnegneu_one(t, client)
            cache[t] = lab
            out[i] = lab
        return out

    with ThreadPoolExecutor(max_workers=w) as ex:
        futs = {ex.submit(predict_posnegneu_one, t, client): (i, t) for i, t in todo}
        for fut in as_completed(futs):
            i, t = futs[fut]
            try:
                lab = fut.result()
            except Exception:
                lab = "neu"
            cache[t] = lab
            out[i] = lab

    return out
