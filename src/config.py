"""Parameters used across the replication of Chen (2026), "Wasserstein
penalisation as an endogenous breadth constraint in penalised mean-variance
portfolio selection".

Every module-level name below is a plain value, grouped by the experiment that
uses it. The comment on each line gives its meaning, unit, and the paper
section it comes from, or states that the paper does not specify it.
"""

# ---------------------------------------------------------------------------
# Shared across experiments (Section 6.1)
# ---------------------------------------------------------------------------
RISK_AVERSION_LAMBDA = 10.0  # lambda in U(phi) = phi'mu - lambda*phi'Sigma*phi, Sec 6.1 "lambda = 10 unless stated otherwise"
ROLLING_WINDOW_DAYS = 60  # estimation window length in trading days, Sec 6.1
N_ASSETS_PANELS = 255  # number of A-share names per panel, Sec 6.1
PANEL_LENGTH_DAYS = 648  # trading days per panel, Sec 6.1
WINDOWS_PER_PANEL = 10  # non-overlapping 60-day estimation windows per 648-day panel (Sec 6.1, 14x10=140 windows of Table 4)
USE_LEDOIT_WOLF = True  # Ledoit-Wolf shrinkage covariance (Ledoit and Wolf, 2004), Sec 6.1 "unless stated otherwise"

HF_DATASET_REPO = "Changahou/GU2026_1"  # authors' public A-share return panels (paper, Data and code availability)
HF_DATASET_REVISION = "main"

# Chronological order of the 14 panel directories, taken from the dataset's
# own manifest.json (https://huggingface.co/datasets/Changahou/GU2026_1).
PANEL_NAMES = [
    "Stock_CN_2018-03-01_2020-10-29",
    "Stock_CN_2018-05-31_2021-01-25",
    "Stock_CN_2018-08-27_2021-04-28",
    "Stock_CN_2018-11-28_2021-07-28",
    "Stock_CN_2019-03-04_2021-11-01",
    "Stock_CN_2019-06-03_2022-01-26",
    "Stock_CN_2019-08-28_2022-05-05",
    "Stock_CN_2019-11-29_2022-08-01",
    "Stock_CN_2020-03-04_2022-11-02",
    "Stock_CN_2020-06-03_2023-02-03",
    "Stock_CN_2020-08-31_2023-05-05",
    "Stock_CN_2020-12-02_2023-08-02",
    "Stock_CN_2021-03-05_2023-11-03",
    "Stock_CN_2021-06-04_2024-01-30",
]

# ---------------------------------------------------------------------------
# Experiment 1: digital verification of Theorem 3.1 (Table 3)
# ---------------------------------------------------------------------------
EXP1_N_ASSETS = 20  # Table 3, N = 20
EXP1_KAPPA0 = 1e-4  # Table 3, kappa_0 = 1e-4 held fixed
EXP1_V_GRID = [1e-3, 1e-2, 1e-1, 1e0]  # Table 3, v values

# ---------------------------------------------------------------------------
# Experiment 2: structural quantities on fourteen A-share panels (Table 4)
# ---------------------------------------------------------------------------
EXP2_N_PANELS = 14  # Table 4, "fourteen A-share panels"
EXP2_N_PANELS_RUN = 14  # executed panel count (identical to the paper; the download is small, see README)
SUPPORT_SIZE_TOL = 1e-3  # weight threshold to count a holding as active; not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 3: radius as a breadth dial on simulated data (Figure 1)
# ---------------------------------------------------------------------------
EXP3_N_ASSETS = 255  # Figure 1 caption, N = 255
EXP3_T_OBS = 60  # Figure 1 caption, T = 60
EXP3_LAMBDA = 10.0  # Figure 1 caption, lambda = 10
EXP3_SEED = 0  # not specified in the paper: fixes the simulated panel
EXP3_N_FACTORS = 5  # not specified in the paper for Fig. 1; reuses the Sec 6.4 factor DGP with no sparse alpha
EXP3_DELTA_GRID = [
    1e-9, 2e-9, 5e-9,
    1e-8, 2e-8, 5e-8,
    1e-7, 2e-7, 5e-7,
    1e-6, 2e-6, 5e-6,
    1e-5, 2e-5, 5e-5,
    1e-4, 2e-4, 5e-4,
    1e-3,
]  # Figure 1, delta grid from 1e-9 to 1e-3

# ---------------------------------------------------------------------------
# Experiment 4: calibration ratio a = sqrt(delta)/s(mu_hat) (Figure 2)
# ---------------------------------------------------------------------------
EXP4_DELTA_GRID = [1e-8, 1e-6, 1e-5, 1e-4]  # Figure 2, four delta values

# ---------------------------------------------------------------------------
# Experiment 5: three scalarisations on the same windows (Table 5)
# ---------------------------------------------------------------------------
EXP5_N_PANELS = 6  # Sec 6.3, "the ex ante audit of Table 1 and the three-form comparison of Table 5 use the first six panels"
EXP5_HOLD_DAYS = 10  # Sec 6.3, "a ten-day holding period"
EXP5_DELTA = 1e-6  # Table 5, delta = 1e-6
EXP5_LAMBDA = 10.0  # Sec 6.1 default

# ---------------------------------------------------------------------------
# Experiment 6: ex ante audit of the breadth cap (Table 1)
# ---------------------------------------------------------------------------
EXP6_N_PANELS = 6  # Sec 6.3, first six panels
EXP6_HOLD_DAYS = 10  # Sec 6.3, ten-day holding period
EXP6_DELTA_GRID = [1e-6, 1e-4]  # Table 1 columns
EXP6_NEFF_TARGETS = [20, 34, 50]  # Table 1, absolute N_eff targets
EXP6_TRACKING_RULES = ["mean_past", "median_past", "previous", "last5", "last10"]  # Table 1 rows
EXP6_LAMBDA = 10.0
EXP6_NW_LAG = max(4 * EXP6_HOLD_DAYS, 10)  # Sec 6.1, Newey-West lag = max(4*HOLD, 10)
EXP6_BOOTSTRAP_B = 2000  # Sec 6.1, stationary bootstrap draws
EXP6_BOOTSTRAP_BLOCK = 20  # Sec 6.1, expected block length

# ---------------------------------------------------------------------------
# Experiment 7: rent of robustness under four benchmarks (Table 6)
# ---------------------------------------------------------------------------
EXP7_N_PANELS = 14  # Sec 6.3, "the four-benchmark comparison of Table 6 ... use all fourteen panels"
EXP7_HOLD_DAYS = 5  # Sec 6.3, five-day holding period
EXP7_DELTA_GRID = [1e-6, 1e-5, 1e-4]  # Table 6 columns
EXP7_LAMBDA = 10.0
EXP7_NW_LAG = max(4 * EXP7_HOLD_DAYS, 10)
EXP7_BOOTSTRAP_B = 2000
EXP7_BOOTSTRAP_BLOCK = 20

# ---------------------------------------------------------------------------
# Experiment 8: disjoint subperiods with transaction costs (Table 7)
# ---------------------------------------------------------------------------
# The two disjoint continuous panels named in Sec 6.3 (2018-03 to 2020-10 and
# 2021-06 to 2024-01) are, respectively, the first and the last of the
# fourteen panel directories shipped with the dataset (see PANEL_NAMES above
# and the dataset manifest.json), so no further slicing is required.
EXP8_PANEL_A = PANEL_NAMES[0]
EXP8_PANEL_B = PANEL_NAMES[-1]
EXP8_HOLD_DAYS = 5  # Sec 6.3, "amortized over the five-day holding period"
EXP8_DELTA_GRID = [1e-6, 1e-5, 1e-4, 1e-3]  # Table 7 columns
EXP8_LAMBDA = 10.0
EXP8_TC_BPS = 30  # Sec 6.3, "turnover costs of thirty basis points"
EXP8_NW_LAG = max(4 * EXP8_HOLD_DAYS, 10)
EXP8_BOOTSTRAP_B = 2000
EXP8_BOOTSTRAP_BLOCK = 20

# ---------------------------------------------------------------------------
# Experiment 9: known-truth factor model simulation (Section 6.4)
# ---------------------------------------------------------------------------
EXP9_N_ASSETS = 255  # Sec 6.4
EXP9_T_OBS = 60  # Sec 6.4
EXP9_LAMBDA = 10.0  # Sec 6.4
EXP9_N_FACTORS = 5  # Sec 6.4, "five factors"
EXP9_ALPHA_FRACTION = 0.2  # Sec 6.4, "sparse alpha ... on 20% of assets"
EXP9_ALPHA_SCALE_C = 2.0  # Sec 6.4 gives the form c*sigma_i/sqrt(T) but not the constant c; not specified in the paper, assumed
EXP9_N_REPS_PAPER = 300  # Sec 6.4, "across 300 replications"
EXP9_N_REPS_RUN = 60  # executed replication count, reduced to fit the time available; see results/RESULTS.md
EXP9_DESIGN_GRID_PAPER = [
    (n, t, lam)
    for n in (255,)
    for t in (40, 60, 80, 120)
    for lam in (5.0, 10.0, 20.0, 30.0)
]  # 16 configurations; the paper states only that N, T and lambda are varied across "16 configurations in a design grid" without enumerating them (Sec 6.4); this grid is an assumed, documented choice
EXP9_DESIGN_GRID_RUN = [
    (255, 60, 10.0),
    (255, 60, 30.0),
    (255, 40, 10.0),
    (255, 120, 10.0),
]  # reduced grid actually executed, see results/RESULTS.md
EXP9_VALIDATION_HOLD_FRACTION = 0.5  # not specified in the paper: split of T used to validate the penalised-robust radius
EXP9_VALIDATION_DELTA_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]  # candidate radii scanned when selecting delta by validation
EXP9_SEED = 1234
