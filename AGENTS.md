# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/nuchad` with subpackages: `analysis/`, `data_processing/`, `visualization/`, `utils/`.
- Script entry points are in `src/scripts/` and are runnable via `python -m scripts.<name>` (see pyproject `project.scripts`).
- Tests reside in `src/tests/` and follow `test_*.py` naming.
- Data inputs live in `data/` (not versioned); outputs are written to `results/`.
- JSON filter configs are in `filtering_configs/`; docs are under `docs/`.

## Build, Test, and Development Commands
- Environment (Python 3.12): `uv add -e .` and for dev tools `uv add -e ".[dev]"` (or `pip install -e .[dev]`).
- Run CLI: `python -m nuchad --task eda` (also: `table1`, `table2`, `table1_stratified`, `table2_stratified`, `visualize`, `reweight`, `filter`, `compare`).
- Run individual scripts: `python -m scripts.make_table1`, `python -m scripts.run_eda`, etc.
- Tests: `pytest -xvs src/tests/` or `./run_tests.sh` (creates `results/` and skips if `data/random_nuchad.csv` is missing).
- Build wheel/sdist: `uv build` (PEP 517 via `hatchling`).
- Lint/format (optional but encouraged): `ruff check src/`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and type hints where reasonable.
- Use snake_case for modules/functions, PascalCase for classes, UPPER_CASE for constants.
- Prefer module-level docstrings and short, purpose-first comments.
- Avoid hard-coded paths; use `nuchad.utils` (`get_data_file`, `get_results_dir`) and write outputs under `results/`.
- Minimize circular imports; defer imports inside functions when needed (pattern used in this repo).

## Testing Guidelines
- Use `pytest`; place tests under `src/tests/` named `test_*.py`.
- Target new or changed logic with focused tests; avoid network and external side effects.
- If tests write artifacts, put them in `results/` with a clear prefix (e.g., `test_...`) and clean up when practical.
- Many tests assume `data/random_nuchad.csv`; skip gracefully when absent (pattern exists in current tests).

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope (e.g., "Add KM curve helper").
- PRs should include: summary, rationale, commands to reproduce (e.g., `python -m nuchad --task table1`), and pointers to generated files in `results/`.
- Link related issues; keep changes small and cohesive; update README/docs when changing CLI or outputs.

## Security & Configuration Tips
- Do not commit datasets or PHI. Keep `data/` untracked (current `.gitignore` excludes `random_nuchad.csv`; avoid adding other data files).
- Be mindful of `stroke_1Y` encoding differences across datasets; run `python analyze_stroke_encoding.py` when in doubt.
- Store filter presets as JSON in `filtering_configs/` and reference by stem (e.g., `--config my_preset`).
