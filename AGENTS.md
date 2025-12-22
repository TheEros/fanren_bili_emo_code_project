# Repository Guidelines

## Project Structure & Module Organization
- `run_pipeline.py` is the main entrypoint; it orchestrates data loading, cleaning, labeling, and reporting.
- `src/` contains core modules: text normalization, feature extraction, lexicon/model emotion tagging, functional tagging, and analysis utilities.
- `manifest.csv` lists per-episode input paths; `emo_lexicon.csv` provides the emotion lexicon.
- `outputs/` is generated and includes `clean/`, `labeled/`, `tables/`, and `figs/` artifacts.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs Python dependencies.
- `python run_pipeline.py --manifest manifest.csv --lexicon emo_lexicon.csv --outdir outputs` runs the full pipeline.
- Optional Ollama flow:
  - `ollama serve` starts the local server.
  - `ollama pull qwen2.5:7b-instruct` downloads a model.
  - `python run_pipeline.py --use_ollama --ollama_model qwen2.5:7b-instruct --ollama_workers 8` enables model-based sentiment.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and snake_case for modules and functions.
- Prefer small, pure helper functions in `src/` with explicit inputs/outputs.
- Keep data columns stable; add new columns with clear, lowercase names (e.g., `model_emo`, `reply_count`).
- No formatter or linter is configured; keep diffs tight and readable.

## Testing Guidelines
- No automated test suite is present.
- When changing logic, validate by running the pipeline on a small manifest and spot-check:
  - `outputs/tables/*_basic_stats.json`
  - `outputs/tables/*_top_terms_*.csv`
- If you add tests, place them under `tests/` and use `pytest` conventions (`test_*.py`).

## Commit & Pull Request Guidelines
- Git history only has an init commit; no established convention.
- Use concise, imperative commit messages (e.g., `Add ollama retry guard`).
- PRs should include: a brief summary, key commands run, and sample output paths (or screenshots of key tables/figs if changed).
