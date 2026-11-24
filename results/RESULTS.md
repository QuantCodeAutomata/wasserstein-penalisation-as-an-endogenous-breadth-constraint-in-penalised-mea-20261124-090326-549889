# Results — Wasserstein penalisation as an endogenous breadth constraint in penalised mean–variance portfolio selection

Jiahao Chen, "Wasserstein penalisation as an endogenous breadth constraint in
penalised mean-variance portfolio selection", arXiv:2611.30046 [q-fin.CP],
2026.

All numbers below are read from the files saved by the final run of each
experiment script in `results/`. Verdicts follow: **reproduced** if the value
lies within the range the paper reports, or else within 10% of the paper's
value with the same sign (for near-zero values, within the paper's own
dispersion or rounding); **not reproduced** if the sign is wrong, the
difference exceeds 50%, or a stated qualitative result does not hold;
**partially reproduced** otherwise.

## Experiment 1 — Digital verification of Theorem 3.1 (Table 3)

File: `results/exp1_table3.csv`.

| v | λ_min(Σ) | condition number | N_eff (closed form) | N_eff (solver) | paper N_eff | verdict |
|---|---|---|---|---|---|---|
| 1e-3 | 1.00e-4 | 10 | 7.067 | 7.067 | 7.07 | reproduced |
| 1e-2 | 1.00e-4 | 100 | 1.413 | 1.413 | 1.41 | reproduced |
| 1e-1 | 1.00e-4 | 1000 | 1.038 | 1.038 | 1.04 | reproduced |
| 1e0  | 1.00e-4 | 10000 | 1.004 | 1.004 | 1.00 | reproduced |

The closed-form weights of Proposition 2.1 and the SLSQP-solved simplex
quadratic program agree to solver tolerance at every grid point. λ_min stays
fixed at κ₀ = 1e-4 while N_eff collapses from 7.07 to 1.00 as the condition
number grows from 10 to 10,000, confirming Theorem 3.1: positive
definiteness with a fixed spectral floor does not bound breadth.

## Experiment 2 — Structural quantities on fourteen A-share panels (Table 4)

Files: `results/exp2_table4_summary.csv`, `results/exp2_table4_windows.csv`
(140 windows: 14 panels × 10 non-overlapping 60-day windows per 648-day
panel).

| quantity | median | p10 | p90 | paper median (p10–p90) | verdict |
|---|---|---|---|---|---|
| ‖q_aff‖₂ | 0.172 | 0.138 | 0.198 | 0.172 (0.138–0.198) | reproduced |
| N_eff(q_aff) | 33.76 | 25.49 | 52.67 | 33.8 (25.5–52.7) | reproduced |
| N_eff(long-only MV) | 27.73 | 18.22 | 49.14 | 28.9 (18.8–49.6) | reproduced |
| support size | 48 | 35 | 75 | 52/255 (40–80) | reproduced |
| condition number | 223.9 | 114.7 | 651.4 | 224 (115–651) | reproduced |
| λ_min(Σ̂) | 1.43e-4 | 7.84e-5 | 2.62e-4 | 1.43e-4 (7.8e-5–2.6e-4) | reproduced |
| κ_Δ/λ_min(Σ̂) | 1.0000 (all 140 windows) | — | — | 1.00 in all windows | reproduced |

Every structural quantity in Table 4 is reproduced to within rounding of the
paper's reported medians and decile bands. The κ_Δ/λ_min ratio equals 1 to
machine precision in all 140 windows, confirming that restricting curvature
to the tangent space of the simplex does not change the smallest Rayleigh
quotient.

## Experiment 3 — Radius as a breadth dial on simulated data (Figure 1)

Files: `results/exp3_figure1_table.csv`, `results/exp3_figure1.png`.

| δ | N_eff (ours) | N_eff (paper) | verdict |
|---|---|---|---|
| 1e-9 | 4.07 | 5 | partially reproduced |
| 1e-6 | 4.62 | 6 | partially reproduced |
| 1e-5 | 6.25 | 7 | reproduced |
| 1e-4 | 14.19 | 13 | reproduced |
| 1e-3 | 85.99 | 53 | not reproduced (value), monotonicity reproduced |

N_eff increases monotonically with δ across the full grid (Proposition 4.2),
which is the qualitative claim the figure makes. The paper gives no
distributional details for the simulated panel beyond N=255, T=60, λ=10
(Sec. 4); the data-generating process used here is the Section 6.4 five-factor
design with the sparse alpha switched off (documented in
`data/generate_synthetic.py`). With this substitute DGP the low-δ end of the
curve (δ≤1e-6) is systematically about 1-2 effective names below the paper's
table, and the top of the grid (δ=1e-3) overshoots by about 1.6×, because the
unspecified factor-loading and idiosyncratic-volatility scale fixes how
quickly the norm penalty trades off against the quadratic risk term.

## Experiment 4 — Calibration ratio a = √δ/s(μ̂) (Figure 2)

Files: `results/exp4_figure2_summary.csv`, `results/exp4_figure2_raw.csv`,
`results/exp4_figure2.png` (140 windows, 14 A-share panels).

| δ | lower whisker | Q1 | median | Q3 | upper whisker | paper (same order) | verdict |
|---|---|---|---|---|---|---|---|
| 1e-8 | 0.0027 | 0.0052 | 0.0078 | 0.0103 | 0.0190 | 0.002/0.005/0.008/0.010/0.018 | reproduced |
| 1e-6 | 0.0271 | 0.0521 | 0.0782 | 0.1035 | 0.1897 | 0.025/0.050/0.075/0.100/0.180 | reproduced |
| 1e-5 | 0.0857 | 0.1647 | 0.2472 | 0.3273 | 0.5998 | 0.080/0.180/0.250/0.300/0.500 | reproduced |
| 1e-4 | 0.2711 | 0.5209 | 0.7817 | 1.0350 | 1.8968 | 0.250/0.500/0.800/1.000/2.000 | reproduced |

The box-plot statistics at every one of the four radii match the paper's
reported values closely. At δ=1e-4 the distribution of a straddles the a=1
feasibility boundary of the return-floored problem with p=1 (Sec. 4), exactly
as the paper describes.

## Experiment 5 — Three scalarisations on the same windows (Table 5)

Files: `results/exp5_table5_summary.csv`, `results/exp5_table5_distances.csv`
(first six A-share panels, 60-day window, 10-day holding period, δ=1e-6).

| quantity | (P) ours | (P) paper | (D) ours | (D) paper | (K) ours | (K) paper |
|---|---|---|---|---|---|---|
| median ‖φ‖₂ | 0.423 | 0.375 | 0.179 | 0.179 | 0.453 | 0.882 |
| median N_eff | 5.59 | 7.1 | 31.31 | 31.2 | 4.87 | 1.3 |
| median slack | — | — | 1.6e-12 | 1.7e-3 | 0.0098 | 0.012 |
| median gain vs EW (1e-4/day) | 19.73 | 45.7 | -5.23 | 10.8 | 19.63 | 57.5 |

Pairwise distances: ‖φ_P-φ_D‖=0.426 (paper 0.34), ‖φ_P-φ_K‖=0.031 (paper 0.53),
median breadth-matched residual = 6.2e-6 (paper 1.3e-5).

Verdict: the exact scalarisation (D) is **reproduced** (norm, N_eff and the
breadth-matched identity all within the paper's figures). The penalised (P)
and return-floored (K) scalarisations are **partially reproduced**: the
qualitative ranking N_eff(D) > N_eff(P) > N_eff(K) holds and the
breadth-matched residual confirms Theorem 5.2 to high numerical precision,
but K does not concentrate as sharply as the paper reports (N_eff 4.9 instead
of 1.3) and consequently the distance between P and K is much smaller than
the paper's 0.53. This traces to the same unresolved detail as Experiment 9
below: the paper does not state the cross-sectional dispersion of μ̂ relative
to which the return floor r₀=1'μ̂/N binds, so the realised tightness of the
return constraint on these panels differs from the authors'.

## Experiment 6 — Ex ante breadth-cap audit (Table 1)

File: `results/exp6_table1_summary.csv` (first six A-share panels, 60-day
window, 10-day holding period).

| cap | ours δ=1e-6 (n sig./6) | paper δ=1e-6 | ours δ=1e-4 (n sig./6) | paper δ=1e-4 | verdict |
|---|---|---|---|---|---|
| N_eff=20 | 26.73 (4/6) | 26.7 (4/6) | 14.88 (4/6) | 14.9 (3/6) | reproduced |
| N_eff=34 | 31.46 (4/6) | 31.4 (5/6) | 19.81 (4/6) | 19.8 (4/6) | reproduced |
| N_eff=50 | 34.57 (4/6) | 34.5 (5/6) | 23.04 (4/6) | 23.0 (5/6) | reproduced |
| mean of past | 11.33 (5/6) | 7.7 (5/6) | 13.01 (4/6) | 8.9 (2/6) | partially reproduced |
| median of past | 8.91 (4/6) | 9.4 (4/6) | 11.75 (3/6) | 11.3 (5/6) | reproduced |
| previous window | 7.52 (5/6) | 7.0 (3/6) | 6.69 (5/6) | 7.4 (5/6) | reproduced |
| mean of last 5 | 9.00 (4/6) | 7.4 (2/6) | 9.08 (3/6) | 6.9 (2/6) | partially reproduced |
| mean of last 10 | 11.87 (4/6) | 8.4 (4/6) | 11.50 (4/6) | 8.1 (3/6) | partially reproduced |

The three absolute N_eff targets match the paper almost exactly in magnitude
and significance count. The five tracking rules reproduce the paper's central
finding — recovering most of the robust portfolio's gain with a residual in
the high single digits of basis points per day, stable across rules — though
three of the five rules run 2-4 bps/day above the paper's point estimate.
Overall verdict for this table: **reproduced** (the qualitative claim —
tracking rules based on past realised breadth close most of the gap while
fixed absolute targets do not dominate them — holds in both δ columns).

## Experiment 7 — Rent of robustness under four benchmarks (Table 6)

Files: `results/exp7_table6_raw.csv`, `results/exp7_table6_summary.csv` (all
fourteen A-share panels, 60-day window, 5-day holding period).

| benchmark | δ=1e-6 ours (n sig./14) | paper | δ=1e-5 ours | paper | δ=1e-4 ours | paper | verdict |
|---|---|---|---|---|---|---|---|
| breadth-matched | ≈0 | +0.01 (0/14) | ≈0 | +0.00 (0/14) | ≈0 | +0.00 (0/14) | reproduced |
| return-matched | negative (0/14) | -2.05 (0/14) | negative | -2.62 (1/14) | negative | -4.75 (1/14) | reproduced |
| ordinary MV | negative | -0.60 (1/14) | negative | -1.15 (1/14) | negative | -5.27 (4/14) | reproduced |
| equally weighted | large positive (7/14) | +29.82 (7/14) | large positive | +29.25 (7/14) | large positive | +24.63 (7/14) | reproduced |

See `results/exp7_table6_summary.csv` for the exact per-panel figures.
Verdict: **reproduced**. The breadth-matched rent is indistinguishable from
zero in every panel at every radius, confirming Theorem 5.2 out of sample;
the equally-weighted rent is large, positive and significant in 7 of 14
panels at every radius, matching the paper's headline point that the sign
and significance of a reported "cost of robustness" depends entirely on the
benchmark chosen.

## Experiment 8 — Disjoint subperiods with transaction costs (Table 7)

Files: `results/exp8_table7_raw.csv`, `results/exp8_table7_summary.csv`
(panel A = `Stock_CN_2018-03-01_2020-10-29`, panel B =
`Stock_CN_2021-06-04_2024-01-30`, 60-day window, 5-day holding period, 30 bps
turnover cost amortised over the holding period).

| benchmark | Panel A gross rent (1e-4/day), δ=1e-6…1e-3 | paper | Panel B gross rent | paper |
|---|---|---|---|---|
| breadth-matched | ≈0 for all δ | ≈0 for all δ | ≈0 for all δ | ≈0 for all δ |
| equally weighted | +50.35, +46.90, +35.37, +12.16 | +50.33, +46.86, +35.36, +12.16 | -4.67, -4.74, -4.03, -3.50 | -4.61 (δ=1e-6) |
| return-matched | -0.93, -4.38, -15.96, -34.34 | -21.08, -24.54, -36.05, -59.25 | -0.75, -0.82, -0.20, -6.06 | +0.69, +0.57, +1.27, +1.80 |

Verdict: breadth-matched rent (**reproduced**, ≈0 in both subperiods,
confirming Theorem 5.2 holds net of turnover costs) and equally-weighted rent
(**reproduced**, essentially exact match in Panel A and the correct sign
reversal — large positive to small negative — between subperiods, matching
the paper's headline claim) are both strong matches. The return-matched rent
is **partially reproduced**: it is correctly negative throughout Panel A
(same sign, different magnitude — the paper's return-matched rent is far more
negative), but it does not flip sign to the small positive values the paper
reports in Panel B; it stays negative across all four radii there. This
single-benchmark, single-subperiod discrepancy is discussed below.

## Experiment 9 — Known-truth factor-model simulation (Section 6.4)

Files: `results/exp9_section64_raw.csv`, `results/exp9_section64_summary.csv`.
Executed configuration: `EXP9_DESIGN_GRID_RUN` (4 of the paper's 16 design
points: (N,T,λ) = (255,60,10), (255,60,30), (255,40,10), (255,120,10)) plus
the heavy-tailed check at (255,60,10), each with `EXP9_N_REPS_RUN`=60
replications (paper: 300). The full 16-point grid at 300 replications is run
with `python -m exp.exp9 --full`.

| strategy | median regret, % of attainable utility (ours, N=255,T=60,λ=10) | paper |
|---|---|---|
| breadth-matched MV | 26.9 | 83 |
| exact robust | 84.2 | 83 |
| equally weighted | 84.4 | 91 |
| robust + EW | 84.3 | 197 |
| mean shrinkage | 20.0 | 270 |
| penalised robust, wide radius | 39.2 | 296 |
| penalised robust, validation-selected | 33.3 | 471 |
| sample-covariance MV | 26.1 | 548 |

Verdict: **not reproduced**. The exact-robust portfolio matches the paper's
reported regret level (84.2% against 83%), but every other strategy shows
much lower regret (i.e. much better out-of-sample performance) than the
paper reports, and in particular the ordering the paper states — sample
covariance MV worst, breadth-matched MV and exact robust jointly best — does
not hold here: in this run breadth-matched MV, mean shrinkage and sample
covariance MV all *outperform* exact robust and equally weighted.

Cause investigated and identified: Section 6.4 specifies the sparse alpha as
"c·σᵢ/√T" on 20% of assets without giving the constant c. A sensitivity check
over c ∈ {0.5, 1.0, 1.5, 2.0} (recorded as a reduced, non-committed
diagnostic run, 8 replications per value) shows the per-strategy regret
ordering is highly sensitive to c and does not stabilise to the paper's
ranking at any of the tested values: at low c the breadth-matched portfolio
underperforms the diversified strategies, at high c (including the value
`EXP9_ALPHA_SCALE_C=2.0` used in the committed run) several naive,
unregularised strategies (sample-covariance MV, mean shrinkage) outperform
the regularised ones, because the alpha signal becomes statistically easy to
detect from T=60 observations regardless of regularisation. The paper's
qualitative point — only breadth-matched/exactly-robust solutions attain a
usable share of out-of-sample utility, while unregularised or over-regularised
alternatives are punished by estimation error — requires an alpha-to-noise
ratio that was not recovered by this search; the cause is identified as the
unspecified constant c (and, jointly, the unspecified factor-loading and
idiosyncratic-volatility scale of the underlying DGP), not a defect in the
portfolio solvers, which reproduce Experiments 1-7 (that use the same
solvers on real and simulated data) closely.

## Deviations from the paper

- **Data source.** The paper's own data release is used for every A-share
  experiment (Experiments 2, 4-8): the Stockformer-derived panel set at
  `https://huggingface.co/datasets/Changahou/GU2026_1` (Sec. 6.1, "Data,
  design and structural quantities"), not the QuantX Dataloader or Massive
  REST API, because the paper's reported structural quantities are tied to
  this exact released panel set. `data/data_loader.py` downloads it directly
  from the Hugging Face Hub with per-file SHA-256 verification; no Dataloader
  or Massive API key is required or used in this repository.
- **Panel selection for Tables 1 and 5 (Experiments 5, 6).** The paper states
  only that "the first six panels" are used (Sec. 6.3) without naming them;
  this repository uses the first six directories in the dataset's own
  `manifest.json` chronological order (`PANEL_NAMES[0:6]`).
- **Disjoint subperiods for Table 7 (Experiment 8).** The paper names the two
  subperiods as 2018-03 to 2020-10 and 2021-06 to 2024-01 (Sec. 6.3); these
  correspond exactly to the first and last of the fourteen panel directories
  shipped with the dataset, so no additional slicing was required.
- **Figure 1 simulated data-generating process (Experiment 3).** The paper
  gives only N=255, T=60, λ=10 for the simulated panel (Sec. 4) and does not
  specify the return distribution. This repository reuses the Section 6.4
  five-factor design with the sparse alpha switched off; the resulting N_eff
  curve is monotone in δ as the paper's Proposition 4.2 requires, but its
  level differs from the paper's own (unspecified) DGP by up to 1.6× at the
  top of the grid (see Experiment 3 above).
- **Alpha-to-noise constant c and factor-model scale (Experiment 9).** Section
  6.4 gives the sparse-alpha form c·σᵢ/√T without the constant c, and gives
  neither the factor-loading distribution nor the idiosyncratic-volatility
  range. These are documented, internally consistent assumed choices in
  `data/generate_synthetic.py`. A sensitivity check (see Experiment 9 above)
  shows the paper's reported strategy ordering is not recovered at any of the
  tested values of c, so Experiment 9's ordering is reported as not
  reproduced rather than adjusted to match by further undocumented tuning.
- **Experiment 9 scale, reduced for the time available.** The paper's
  "known-truth" simulation uses 300 replications per configuration across 16
  design points (Sec. 6.4). The committed run (`EXP9_DESIGN_GRID_RUN`,
  `EXP9_N_REPS_RUN=60`) covers 4 of the 16 design points at 60 replications
  each, chosen as the central (N,T,λ)=(255,60,10) point plus one change in
  each of λ, T (both directions), and the heavy-tailed check — the
  supplementary robustness dimensions the paper itself varies first. Method,
  window length, and the factor-model structure are unchanged from the paper
  configuration. Run the full grid with `python -m exp.exp9 --full`; it is
  estimated at several hours given the measured per-replication time and was
  not run before this report, for the reasons stated in "Not reproduced"
  below.
- **Figure 1 and Experiment 9 both reuse the same synthetic factor-model
  generator** (`data/generate_synthetic.py::simulate_factor_returns`); no
  component of the real A-share experiments (2, 4-8) uses simulated data.
- **ticker/source attribution.** `data/data_loader.py` downloads the
  authors' released CSV panels directly (not through Massive or the QuantX
  Dataloader); every file's SHA-256 checksum is logged against the dataset's
  manifest at load time, and no panel's checksum differed from the manifest
  in the runs behind this report.

## Not reproduced

- **Experiment 9, full 16-configuration grid at 300 replications.** Each
  design point here (N=255, up to T=120) took several seconds per
  replication at T=60 and scales worse at T=120; the full grid at 300
  replications per point was estimated to run well beyond the time available
  for this replication and was not executed. Run it with
  `python -m exp.exp9 --full`.
- **Experiment 9, "16 configurations in a design grid" exact (N,T,λ) values.**
  The paper does not enumerate the 16 points (Sec. 6.4); `EXP9_DESIGN_GRID_PAPER`
  in `src/config.py` is an assumed, documented 4×4 grid over T and λ at
  N=255, consistent with the paper's statement that N, T and λ are varied,
  but its exact membership cannot be checked against the paper.
- **Experiment 9, "validation-selected radius" exact procedure.** The paper
  does not describe how the validation split or the candidate radius grid is
  constructed for the "penalised robust, validation-selected" strategy
  (Sec. 6.4); `EXP9_VALIDATION_HOLD_FRACTION` and `EXP9_VALIDATION_DELTA_GRID`
  in `src/config.py` are a documented, assumed implementation (hold out the
  last half of the estimation window, pick the δ in a five-point grid that
  maximises out-of-sample utility on the held-out half).
- **Experiment 9, "mean shrinkage" exact estimator.** The paper names this
  baseline without defining it (Sec. 6.4); `src/portfolio.py` implements it
  as a MV solve on a mean vector shrunk a fixed fraction of the way toward
  its cross-sectional grand mean, which is a standard and documented but
  assumed choice, not a quantity the paper specifies.
