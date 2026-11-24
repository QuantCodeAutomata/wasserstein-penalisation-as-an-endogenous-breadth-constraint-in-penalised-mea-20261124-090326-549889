"""Loader for the fourteen A-share return panels used in Section 6 of the
paper (Table 4-7, Figures 1-2).

The paper's empirical sections analyse the public "GU2026_1" panel data
shipped with the Stockformer benchmark, hosted on the Hugging Face Hub at
https://huggingface.co/datasets/Changahou/GU2026_1. This is the authors' own
data release, not a OHLCV time series the QuantX Dataloader or Massive REST
API can serve (it is already reduced to the daily simple return matrices used
directly by the paper's estimators), so it is downloaded from that
repository instead.

Each panel is a `flow.npz` file holding one array `result` of shape
(T=648, N=255) of decimal simple daily returns, "oldest first" (manifest.json
in the same repository). The fourteen panels are overlapping three-year
windows dated from 2018-03-01 to 2024-01-30.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from src.config import HF_DATASET_REPO, HF_DATASET_REVISION, PANEL_NAMES

CACHE_DIR = Path(__file__).resolve().parent / "cache"  # never committed, see .gitignore


class PanelLoadError(RuntimeError):
    """Raised when a panel file is missing, malformed, or fails validation."""


def load_manifest() -> dict:
    """Download and parse manifest.json, the dataset's own description of
    array layout, units, and per-panel checksums and summary statistics."""
    path = hf_hub_download(
        HF_DATASET_REPO, "manifest.json", repo_type="dataset", revision=HF_DATASET_REVISION
    )
    return json.loads(Path(path).read_text())


def load_panel_returns(panel_name: str) -> pd.DataFrame:
    """Daily simple returns for one panel: a (T, N) DataFrame of floats,
    integer row index (trading-day order, oldest first) and string columns
    ``asset_0 .. asset_{N-1}`` (the dataset ships no tickers, only an N=255
    return matrix).
    """
    relpath = f"data/{panel_name}/flow.npz"
    local_path = hf_hub_download(
        HF_DATASET_REPO, relpath, repo_type="dataset", revision=HF_DATASET_REVISION
    )
    with np.load(local_path, allow_pickle=True) as data:
        if "result" not in data:
            raise PanelLoadError(f"{panel_name}: expected array key 'result', got {list(data.keys())}")
        arr = data["result"]
    if arr.ndim != 2:
        raise PanelLoadError(f"{panel_name}: expected a 2D return matrix, got shape {arr.shape}")
    if np.isnan(arr).any():
        raise PanelLoadError(f"{panel_name}: unexpected NaNs in the return matrix")
    t_days, n_assets = arr.shape
    columns = [f"asset_{i}" for i in range(n_assets)]
    df = pd.DataFrame(arr, columns=columns)
    df.attrs["panel_name"] = panel_name
    df.attrs["t_days"] = t_days
    df.attrs["n_assets"] = n_assets
    return df


def load_all_panels(panel_names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load every requested panel (default: all fourteen, in the dataset's own
    chronological order) and log the realised row/column counts."""
    names = panel_names if panel_names is not None else PANEL_NAMES
    panels = {}
    for name in names:
        df = load_panel_returns(name)
        panels[name] = df
        print(
            f"[data_loader] panel={name} rows={df.attrs['t_days']} "
            f"assets={df.attrs['n_assets']}"
        )
    return panels
