# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for SurvivEHR (medRxiv 10.1101/2025.08.04.25332916): a decoder-only transformer foundation model pre-trained on UK primary care EHR (CPRD) with a competing-risks time-to-event objective, then evaluated zero-shot, few-shot or fine-tuned on downstream prognostic tasks. Upstream author is Charles Gadd (University of Oxford); `origin` is a fork. GPL v3.

No model weights and no data are in the repo. Every config and job script points at `/rds/...` paths on the University of Birmingham BlueBEAR HPC (Slurm, `module load`, Apptainer). Nothing here runs end to end without the CPRD SQLite database and FastEHR datasets on that cluster.

## Environment

- Python 3.10 (code uses `match` statements). `pyproject.toml` + `uv.lock` are the uv project (`make sync`, `make lock`, `make build`, `make lint`, `make format`; `make help` lists them). `requirements.txt` / `requirements.lock` are the older pip-style pins still used by `containers/` and the HPC job scripts; direct pins in `pyproject.toml` mirror `requirements.txt`, and `[tool.uv] constraint-dependencies` pins every transitive package to `requirements.lock` (the environment the paper's results were produced with). Change them together. Old stack: `torch==2.0.1`, `pytorch-lightning==2.1.0`, `polars==0.17.12`, `hydra-core==1.3.2`, `wandb==0.13.6`. Don't write code that assumes newer APIs (especially polars).
- **FastEHR** (`https://github.com/cwlgadd/FastEHR`) is a required sibling dependency not listed in `requirements.txt`. It provides `FoundationalDataModule`, the tokeniser, the database builder and the BEHRT adapters. Clone it next to this repo and put it on `PYTHONPATH`.
- **Imports are package-rooted at `SurvivEHR.`** (`from SurvivEHR.src.models...`, `from SurvivEHR.examples.modelling...`). The repo directory must be named `SurvivEHR` and its *parent* must be on `PYTHONPATH`; the top-level `__init__.py` exists for this. The package was renamed from `CPRD` in `ece8753` — the README's `mv SurvivEHR CPRD` step and a few strings (e.g. `_target_="CPRD.examples..."` in `notebooks/CompetingRisk/3_Regional_analaysis/regional_run.py`, `src/README.md`) are stale.
- Apptainer route (HPC): `containers/env.paths.sh` holds all paths; `bash containers/container_build.sh`, `bash containers/env_bootstrap.sh`, then `bash containers/run_in_container.sh python ...` (adds `--nv` automatically when Slurm/CUDA GPUs are visible).

- Packaging: hatchling remaps the repo root to `SurvivEHR/` in the wheel (`sources`), and ships only `*.py` and `*.yaml` from `src/` and `examples/` (notebooks and artefacts excluded). Editable installs (`uv sync`) put the *parent* directory on `sys.path` via `dev-mode-dirs = [".."]`, because hatchling can't do editable installs with a prefix-adding rewrite.
- `make lint` is check-only bug rules (`E4`, `E7`, `E9`, `F`) and currently reports ~255 existing upstream issues. `make format` would rewrite the whole upstream codebase; don't run it without asking.

There is no test suite or CI.

## Running experiments

Everything goes through the Hydra entry point `examples/modelling/SurvivEHR/run_experiment.py`. Its decorator names `config_name="default"`, which doesn't exist, so always pass `--config-name`:

```bash
cd examples/modelling/SurvivEHR
python run_experiment.py --config-name=config_CompetingRisk11M            # or config_SingleRisk
python run_experiment.py --config-name=config_CompetingRisk11M \
    experiment.type=fine-tune-cr experiment.fine_tune_id=FineTune_CVD \
    data.path_to_ds=/path/to/FineTune_CVD/ fine_tuning.fine_tune_outcomes=[...]
```

Notebook sweeps (`notebooks/**/ablation_run.py`, `regional_run.py`) instead use `hydra.initialize/compose` with overrides and call `run(cfg)` directly. Slurm wrappers are the `*_job.sh` files alongside them.

`experiment.type` (normalised by stripping `-`/spaces and lowercasing) selects the path in `run()`:

| type | setup function | what it does |
|---|---|---|
| `pre-train` / `causal` / `selfsupervised` | `setup_causal_experiment` | next-event causal pre-training |
| `zero-shot` | `setup_fewshot_experiment` | evaluate pre-trained head on a supervised task, `train` must be False |
| `few-shot` | `setup_fewshot_experiment` | supervised training keeping the pre-training head |
| `fine-tune` / `fine-tune-cr` / `fine-tune-sr` | `setup_finetune_experiment` | replace head with a new CR (default) or SR survival head |

The data module is loaded in supervised form when `fine_tuning.fine_tune_outcomes` or `fine_tuning.custom_outcome_method._target_` is set. Custom outcome and stratification methods are given as dotted `_target_` strings and resolved with `importlib` (see `helpers.custom_mm_outcomes`, `examples/data/study_criteria.py`).

**Checkpoint resolution is implicit and path-based**: pre-trained = `{ckpt_dir}{run_id}.ckpt`, supervised = `{ckpt_dir}{run_id}_{fine_tune_id}.ckpt`. If the supervised checkpoint exists it is loaded for evaluation (and `train` must be False); else if the pre-trained one exists it initialises from it; else it trains from scratch. Continuing pre-training writes to a `-v1` checkpoint. After `trainer.fit` the best checkpoint is reloaded before `trainer.test`. Logging is W&B (`WandbLogger`).

`run()` also mutates config at runtime: `cfg.head.tokens_for_univariate_regression` is filled from the data module's `meta_information["measurement_tables"]` (measurements with observations get a Gaussian value head).

## Architecture

Library code is in `src/`; experiment wiring (Lightning modules, optimisers, schedulers, callbacks) is in `examples/modelling/SurvivEHR/`.

- **Backbone** — `src/models/TTE/base.py::TTETransformer`. `DataEmbeddingLayer` (`src/modules/data_embeddings/`) embeds each event from token + optional value + static covariates; `TemporalPositionalEncoding` encodes event *time*, not sequence index. Blocks are Neo (`global`/`local` attention) or Nano GPT variants, chosen by `transformer.block_type`. `private_heads` reserves some attention-head dimensions for the value head only.
- **Pre-training model** — `src/models/survival/task_heads/causal.py::SurvStreamGPTForCausalModelling`. Survival head from `head.SurvLayer`: `cr` → `ODESurvCompetingRiskLayer` (next event as competing risks over the whole vocabulary, the paper's core idea); `sr` → `CausalODESurvSingleRiskLayer` (one-vs-all). Both are DeSurv-style ODE survival layers built on `src/modules/head_layers/survival/desurv.py`. Plus `GaussianRegressionLayer` for measurement values. Loss is `surv_weight`/`value_weight`, normalised to sum to 1.
- **Lightning wrappers** — `CausalExperiment`, `FewShotExperiment`, `FineTuneExperiment` in `setup_*_experiment.py`. Fine-tuning supports PEFT (`fix`/`adapter`), layer-wise LR decay, linear probing then top-k unfreezing, and an optional compression layer (`fine_tuning.*` config; optimiser groups in `optimizers_fine_tuning.py`). The custom `DecayCAWarmRestarts` scheduler lives in `setup_causal_experiment.py`.
- **Callbacks** — `src/models/base_callback.py` (UMAP hidden-embedding plots) and `src/models/survival/custom_callbacks/` (causal eval metrics, clinical-prediction metrics such as C-td/IBS/INBLL, RMST).
- **Legacy** — `src/models/transformer/` (plain GPT with `causal_lm` head) imports `SurvivEHR.src.modules.block`, which no longer exists. `src/models/TTE/task_heads/` are also earlier variants. The live path is `TTE/base.py` + `survival/task_heads/causal.py`.

`examples/data/` is the numbered data pipeline built on FastEHR: 1 build the SQLite DB, 2 build the pre-training dataset (practice-level splits to avoid leakage, meta information for tokenisation and outlier bounds), 3 build fine-tuning datasets / cross-sectional benchmark data, 4 convert to BEHRT format. `examples/modelling/benchmarks/` holds DeepHit, DeSurv, RSF and MLP baselines.

## Notes

- `examples/modelling/SurvivEHR/notebooks/` holds the manuscript experiments, organised by paper section (`CompetingRisk/1_pretraining`, `2_1_Cardiovascular_disease`, …). Many committed PNGs and notebook outputs are generated; don't edit them by hand.
- Commit history uses conventional-commit prefixes (`refactor(imports): ...`, `chore(...)`).
