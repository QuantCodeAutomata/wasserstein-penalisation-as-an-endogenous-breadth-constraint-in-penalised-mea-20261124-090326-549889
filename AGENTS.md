# Repository guide

This repository replicates nine experiments from Jiahao Chen, "Wasserstein
penalisation as an endogenous breadth constraint in penalised mean-variance
portfolio selection", arXiv:2611.30046 [q-fin.CP], 2026: the digital
verification of Theorem 3.1 (Table 3), the structural portfolio quantities on
fourteen A-share panels (Table 4), the radius-as-breadth-dial illustration
(Figure 1), the calibration of the Wasserstein radius (Figure 2), the
comparison of the three Wasserstein scalarisations (Table 5), the ex ante
breadth-cap audit (Table 1), the rent of robustness under four benchmarks
(Table 6), the disjoint-subperiod check with transaction costs (Table 7), and
the known-truth factor-model simulation (Section 6.4).

## Layout

- `src/` — flat library modules: `portfolio.py` (all portfolio solves:
  affine and long-only minimum variance, penalised/exact/return-floored
  Wasserstein scalarisations, breadth-matched ridge benchmark, mean
  shrinkage), `windows.py` (rolling-window estimation and holding-period
  alignment), `statistics.py` (Newey-West t-statistics, stationary bootstrap,
  Romano-Wolf step-down), `empirical.py` (shared panel-loading and
  per-window orchestration used by experiments 2 and 4-8), `config.py` (every
  parameter as a module-level `UPPER_CASE` variable with a comment on its
  meaning, unit and paper reference).
- `exp/` — one script per experiment (`exp1.py` … `exp9.py`), run as
  `python -m exp.expN`; each writes its outputs directly into `results/`.
- `data/` — `data_loader.py` downloads and caches the authors' public A-share
  panel release from the Hugging Face Hub; `generate_synthetic.py` builds the
  simulated factor-model data used by experiments 3 and 9. Downloaded or
  generated data is cached under `data/cache/` and `.gitignore`d, never
  committed.
- `tests/` — pytest tests for `src/` components (synthetic/fixture inputs)
  and experiment-level tests. Tests that need real A-share data are marked
  `market_data` and use a session-scoped cached slice loaded through
  `data/data_loader.py`.
- `results/` — tables, figures and `RESULTS.md` written by the experiment
  scripts. Re-running an experiment overwrites its files in `results/`; the
  numbers and verdicts in `results/RESULTS.md` are not regenerated
  automatically and must be checked against the new files and updated by hand
  after a rerun.

## Commands

```bash
python -m pip install -r requirements.txt
python -m pytest -m "not market_data"      # unit tests, no network
python -m pytest                            # also exercises a cached A-share slice
python -m exp.exp1                          # ... through exp9, see README.md
python -m exp.exp9 --full                   # paper's full 16-point, 300-rep grid
./reproduce.sh                              # tests + all nine experiments
```

No API key is required: the A-share data is the authors' own public release,
downloaded directly from the Hugging Face Hub (`data/data_loader.py`); the
QuantX Dataloader and Massive REST API are not used in this repository.

## Conventions

- All experiment parameters live only in `src/config.py` as module-level
  `UPPER_CASE` variables, each with a one-line comment giving its meaning,
  unit and source (a paper section/equation, or "not specified in the
  paper"); no classes or parameter dictionaries. Where a run reduces a
  paper-scale count (replications, grid points) to fit the time available,
  the paper value and the executed value are two adjacent variables (for
  example `EXP9_N_REPS_PAPER` / `EXP9_N_REPS_RUN`), and an experiment's
  `--full` flag selects the paper values.
- Market/panel data is loaded only through `data/data_loader.py`; synthetic
  data only through `data/generate_synthetic.py`. No experiment script or
  `src/` module fetches or generates data on its own.
- Experiments call the shared `src/` functions by their documented argument
  names and return keys; they do not re-implement a solver or a performance
  calculation locally.
- Deviations from the paper (data source, universe, interpretations of
  details the paper leaves open, reduced configurations, and any material
  gap to the paper's reported numbers with its identified cause) are
  recorded only in `results/RESULTS.md`, never as silent changes to the code
  or to `src/config.py`.
