# Changelog

## v4
- 新增 src/emo_model.py：调用本地 Ollama 预测 pos/neg/neu
- run_pipeline.py 增加参数：--use_ollama / --ollama_model / --ollama_base_url / --ollama_workers / --ollama_only_for_other
- label_danmaku / label_comment 支持可选模型情绪，并与 lex_emo 通过 combine_emo 融合
