# fanren_bili_emo

多集处理：弹幕/评论清洗、特征抽取(features)、情绪(emo)与功能(func)标签、刷屏检测、基础统计与高频词粗看导出。

## 安装依赖
```bash
pip install -r requirements.txt
```

> 注：高频词粗看默认自带 **fallback 分词**（不依赖 jieba）。如果你本地装了 jieba，会自动优先使用 jieba 结果。

## 运行
```bash
python run_pipeline.py --manifest manifest.csv --lexicon emo_lexicon.csv --outdir outputs
```

## 关键输出（对应你论文步骤）
- **步骤2-2 弹幕基础统计**：`outputs/tables/ep{ep}_danmaku_basic_stats.json`
- **3.2.1 词表来源：高频词粗看**：
  - `outputs/tables/ep{ep}_top_terms_danmaku.csv`
  - `outputs/tables/ep{ep}_top_terms_comment.csv`
- 其余：emo/func 分布、minute 曲线、2秒窗刷屏、Top根楼(高赞/高回复)等，详见 tables/。


## 使用 Ollama 本地模型做 model_emo（pos/neg/neu）
1) 安装并启动 Ollama
```bash
ollama serve
```
2) 拉取模型（示例）
```bash
ollama pull qwen2.5:7b-instruct
```
3) 运行 pipeline（推荐只对 lex_emo=other 调用模型）
```bash
python run_pipeline.py --use_ollama --ollama_model qwen2.5:7b-instruct --ollama_workers 8
```
若不想用模型，直接不加 --use_ollama 即可。


### Ollama 输出包含 <think> 或长解释怎么办？
- emo_model.py 已设置 think=false，并使用 format(JSON Schema) 强约束输出为 {"label":"pos|neg|neu"}。
- 同时对输入中的 /think 做了中和，避免模式注入。
- 若仍不稳定，建议换成 qwen2.5:7b-instruct 等更听话的 instruct 模型。


## 排查：为什么 model_emo 全是 neu？
1) 先确认 Ollama API 能访问（同一环境里）：
```bash
curl http://localhost:11434/api/tags
```
2) 开启调试（会打印少量错误与返回片段）：
```bash
# Linux/macOS
OLLAMA_DEBUG=1 python run_pipeline.py --use_ollama --ollama_model qwen3:8b --ollama_workers 4
# Windows PowerShell
setx OLLAMA_DEBUG 1
```
3) 如果你在 Docker/WSL 里跑 Python，而 Ollama 跑在宿主机：
- Docker：--ollama_base_url http://host.docker.internal:11434
- WSL：通常仍是 http://localhost:11434，但若不通可用宿主机 IP
