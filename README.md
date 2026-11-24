# Wasserstein penalisation as an endogenous breadth constraint in penalised mean-variance portfolio selection

Replication of: Jiahao Chen, "Wasserstein penalisation as an endogenous breadth
constraint in penalised mean-variance portfolio selection", arXiv:2611.30046
[q-fin.CP], 2026.

This repository reproduces nine experiments from the paper: the digital
verification of the impossibility theorem (Theorem 3.1, Table 3), the
structural portfolio quantities measured on fourteen A-share panels (Table 4),
the radius-as-breadth-dial illustration on simulated data (Figure 1), the
calibration of the Wasserstein radius (Figure 2), the comparison of the three
Wasserstein scalarisations (Table 5), the ex ante breadth-cap audit (Table 1),
the rent of distributional robustness under four benchmarks (Table 6), the
disjoint-subperiod check with transaction costs (Table 7), and the
known-truth factor-model simulation (Section 6.4).

## Layout

```
src/     portfolio construction, rolling-window pipeline, inference tools, config
exp/     one script per experiment, run as a module
data/    data_loader.py (A-share panels) and generate_synthetic.py (simulations)
tests/   pytest unit tests and A-share-panel smoke tests
results/ tables, figures and RESULTS.md written by the experiments
```

## Setup

```bash
python -m pip install -r requirements.txt
```

No API key is required: the A-share panels are the authors' own public data
release (see `data/data_loader.py`), downloaded directly from the Hugging Face
Hub at `https://huggingface.co/datasets/Changahou/GU2026_1`.

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests, no network
python -m pytest                         # also downloads a reduced A-share slice
```

## Experiments

Each experiment is run as a module from the repository root and writes its
outputs directly into `results/`:

```bash
python -m exp.exp1   # Table 3:  positive definiteness does not imply breadth
python -m exp.exp2   # Table 4:  structural quantities on 14 A-share panels
python -m exp.exp3   # Figure 1: radius as a breadth dial (simulated data)
python -m exp.exp4   # Figure 2: calibration of the Wasserstein radius
python -m exp.exp5   # Table 5:  three scalarisations on the same windows
python -m exp.exp6   # Table 1:  ex ante breadth-cap audit
python -m exp.exp7   # Table 6:  rent of robustness under four benchmarks
python -m exp.exp8   # Table 7:  disjoint subperiods with transaction costs
python -m exp.exp9   # Section 6.4: known-truth factor-model simulation
```

Experiments 5-9 download the fourteen A-share panels (or a subset, see
`src/config.py`) the first time they run and cache them under `data/cache/`
(not committed). A `--full` flag, where present, selects the paper's full
configuration (panel count, replication count) rather than the reduced
configuration executed for this repository; see `results/RESULTS.md` for the
executed configuration of every experiment and the paper's own values.

`reproduce.sh` runs the full test and experiment sequence above.

## Results

See `results/RESULTS.md` for every experiment's numbers next to the paper's
reported values, a verdict for each compared metric, and the "Deviations from
the paper" and "Not reproduced" sections.
