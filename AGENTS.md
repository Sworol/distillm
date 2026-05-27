# Repository Guidelines

## Project Structure & Module Organization
- `distillm/`: core distillation implementation (training loops, utilities used by scripts).
- `minillm/`: upstream baseline components this repo builds on (MiniLLM-style training/distillation).
- `data_utils/` and `tools/`: dataset prep, tokenization, and helper scripts (e.g., OpenWebText utilities).
- `scripts/`: runnable experiment entrypoints grouped by model family (`gpt2/`, `opt/`, `openllama2/`) and task (`sft/`, `kd/`, `distillm/`, `eval/`, `tools/`).
- `configs/`: configuration files used by training/eval.
- Top-level entrypoints: `finetune.py`, `generate.py`, `evaluate.py`, `evaluate_main.py`, `train_minillm.py`.

## Build, Test, and Development Commands
- `bash install.sh`: installs Python deps used by training/evaluation (PyTorch is commented; install it separately for your CUDA setup).
- `python3 tools/get_openwebtext.py`: prepares a line-delimited OpenWebText text file for parallel processing.
- Data processing (examples):
  - `bash scripts/gpt2/tools/process_data_dolly.sh /PATH/TO/DistiLLM $MASTER_PORT $GPU_NUM`
  - `bash scripts/gpt2/tools/process_data_pretrain.sh /PATH/TO/DistiLLM $MASTER_PORT $GPU_NUM`
- Training (examples):
  - `bash scripts/gpt2/distillm/train_base_xl.sh /PATH/TO/DistiLLM $MASTER_PORT $GPU_NUM`
- Evaluation:
  - `bash scripts/gpt2/eval/run_eval.sh $GPU_IDX /PATH/TO/DistiLLM`

## Coding Style & Naming Conventions
- Python: 4-space indentation; prefer explicit, descriptive names; keep new utilities in `distillm/` or `tools/` (not scattered in root).
- Filenames: `snake_case.py`; shell scripts under `scripts/**` follow existing folder conventions by model family/task.
- Avoid committing large artifacts (datasets, `checkpoints/`, generated binaries). Prefer documenting paths and using `.gitignore`.

## Testing Guidelines
- This repo does not include a dedicated unit test suite. Validate changes with “smoke runs”:
  - Import check: `python -c "import distillm"`
  - Script dry run on a tiny config/batch before launching multi-GPU jobs.
- When adding new features, include a minimal reproducible command in your PR description.

## Commit & Pull Request Guidelines
- Commit messages in history are short and imperative (e.g., `Update README.md`, `[Update] Remove ...`). Follow that pattern; use `[Update]` for repo-wide behavioral changes.
- PRs should include: what changed, why, exact command(s) used to verify, and any new/updated scripts/configs referenced.
- If changes affect results, note the model family (`gpt2/`, `opt/`, `openllama2/`) and the evaluation command used.

## Security & Configuration Tips
- Do not hardcode credentials or private paths in `scripts/` or `configs/`.
- Keep machine-specific settings (ports, GPU counts, checkpoint locations) as shell args/env vars (see existing scripts).
