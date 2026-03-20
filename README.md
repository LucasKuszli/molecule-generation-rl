
# molecules-and-ai-project

Short guide and developer notes for the repository. This README describes the rough project layout, how to get started (including how to use `uv` to manage dependencies / environments), how to generate the images in `img/`, and a short list of useful scripts that do not require training.

## Quick overview

Top-level layout (high-level):

- `src/` — main Python package with model code, data pipeline, evaluation, and scripts.
- `data/` — data artifacts and prepared training CSVs used by the code.
- `artifacts/` — trained model checkpoints and other downstream artifacts produced by experiments.
- `evaluation/` — evaluation results, summaries and per-seed outputs used to make plots.
- `img/` — generated figures and visualizations (PNG files) created by plotting scripts.
- `scripts/` — small top-level helper scripts (quick demos / explorers).
- `pyproject.toml` — project metadata and dependency declarations. This repo includes a `tool.uv` section so `uv` can be used to manage environments and indexes.

## Requirements

- Python >= 3.11 (project configured in `pyproject.toml`).
- Recommended: install `uv` (fast project manager) to run scripts and manage the virtual environment as shown below. You can also use `pip`/`venv` directly if you prefer.

## Dependency / environment setup (using uv)

This project includes a `tool.uv` index configuration in `pyproject.toml` so `uv` can pick up the alternate package index (for example the PyTorch wheel index). The recommended workflow using `uv` is:

1. Install `uv` (one of the options):

	 - With pipx (recommended):

		 pipx install uv

	 - With pip:

		 pip install --upgrade uv

	 - Or use the official installer script (see https://astral.sh/uv/install.sh).

2. From the repository root, create or refresh the project virtual environment and sync deps. Two common flows:

	 - Let `uv` manage the project environment (preferred):

			 uv venv
			 uv lock        # resolves/locks deps from pyproject
			 uv sync        # syncs the .venv to the locked state

		 After that you can run project commands with `uv run ...` which ensures they run inside the managed environment.

	 - If you don't want to use `uv` to sync, fallback to an editable install using pip:

			 python -m venv .venv
			 source .venv/bin/activate
			 python -m pip install --upgrade pip build
			 pip install -e .

Notes:

- `uv run <cmd>` is used across the repo to run commands inside the environment, e.g. `uv run python -m src.data.setup`.
- Because `pyproject.toml` contains a `tool.uv.index` entry pointing to the PyTorch wheel index, using `uv` ensures the custom index will be used for binary packages.

## Generating images (img/)

Many plotting and visualization scripts write PNGs into the `img/` folder. The general pattern is to run the module with `uv run python -m <module>`.

Examples:

- Generate the molecule panel (samples/representatives from generators + ground-truth):

		uv run python -m src.scripts.plot_molecule_panel

	Output: `img/molecule_panel.png` (this script samples generator checkpoints and therefore expects generator artifacts and the FP evaluator to exist in `artifacts/` — see notes below.)

- Benchmark comparison plot (reads `benchmark/summary.csv`):

		uv run python -m src.evaluation.benchmark_plots

	Output: `benchmark/plots/benchmark_metrics_comparison.png` (requires `benchmark/summary.csv` created by previous evaluation runs).

- Build evaluation report / plots (reads evaluation outputs under `evaluation/`):

		uv run python -m src.evaluation.report

	Output: various images in `img/` (depends on the evaluation artifacts present in the repository).

Notes about artifacts:

- Some plotting scripts require model artifacts (checkpoints, evaluator pickles). Those scripts will fail or produce placeholder outputs if `artifacts/` is missing or incomplete. If you do not want to run training, either copy precomputed artifacts into the `artifacts/` tree or run the data-only scripts below.

## Useful scripts that do NOT require training

The following scripts are helpful for preparing data and generating visualizations and do not require training models. They either download / preprocess data or generate plots from already-produced evaluation files.

- Data-download and preprocessing (no model training):
	- `uv run python -m src.data.download` — download raw datasets from Hugging Face and save local CSVs.
	- `uv run python -m src.data.preprocess` — sanitize SMILES, desalt/neutralise, deduplicate and scaffold-split into train/val/test CSVs.
	- `uv run python -m src.data.setup` — convenience end-to-end data setup (download + preprocess).

- Evaluation / plotting from results (no training required if results exist):
	- `uv run python -m src.evaluation.benchmark_plots` — create benchmark comparison charts from `benchmark/summary.csv`.
	- `uv run python -m src.evaluation.report` — generate the evaluation report and figures from outputs in `evaluation/`.

- Small top-level viewer scripts (note: these require model checkpoints if you want model predictions; without checkpoints they still illustrate the viewer code):
	- `scripts/sample_hiv_actives.py` and `scripts/view_hiv_inactives.py` — display molecules with predicted scores (need `artifacts/gnn/*` checkpoint for predictions). The data reading / plotting code itself runs without training if you provide a saved checkpoint.

If you want to run purely data/visual tasks with zero training, use the `src.data.*` tools to prepare the data and the `src.evaluation.*` plotting utilities to render plots from existing evaluation CSVs under `evaluation/` or `benchmark/`.

## Running (examples)

- Prepare data:

		uv run python -m src.data.setup

- Generate plots (if evaluation / benchmark outputs exist):

		uv run python -m src.evaluation.benchmark_plots
		uv run python -m src.evaluation.report

- Run a small script (opens windows or saves images):

		uv run python scripts/sample_hiv_actives.py

## Notes and next steps

- If you plan to reproduce experiments, check the `artifacts/` layout and pipeline modules inside `src/pipeline/` for end-to-end orchestration.

## License / contact

This project is licensed under the MIT License — see the accompanying `LICENSE` file for details.

Contact the maintainer listed in the repository metadata for questions.


