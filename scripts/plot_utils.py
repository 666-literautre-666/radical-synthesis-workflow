"""
Publication-quality plotting for chemistry data.
All figures are saved in editable vector formats (SVG + PDF) plus raw data (CSV)
so you can tweak them in Illustrator, Inkscape, or Origin.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Sequence
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STYLES_DIR = PROJECT_ROOT / "styles"
FIGURES_DIR = PROJECT_ROOT / "data" / "figures"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
EXPERIMENTAL_DIR = PROJECT_ROOT / "data" / "experimental"

for d in [FIGURES_DIR, PREDICTIONS_DIR, EXPERIMENTAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Journal style registry
# ---------------------------------------------------------------------------
JOURNAL_STYLES = {
    "jacs": STYLES_DIR / "jacs.mplstyle",
    "angewandte": STYLES_DIR / "angewandte.mplstyle",
    "nature_chem": STYLES_DIR / "nature_chem.mplstyle",
    "acs": STYLES_DIR / "acs.mplstyle",
    "rsc": STYLES_DIR / "rsc.mplstyle",
}


def use_journal_style(journal: str = "jacs"):
    """Activate a journal-specific matplotlib style."""
    style_path = JOURNAL_STYLES.get(journal.lower())
    if style_path and style_path.exists():
        plt.style.use(str(style_path))
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


# ---------------------------------------------------------------------------
# Figure context manager — the core API
# ---------------------------------------------------------------------------

@dataclass
class ChemFigure:
    """
    Context manager that produces a publication-ready figure.
    On exit, saves: SVG (editable vector), PDF, PNG, pickle (re-editable in Python),
    and the raw data as CSV.

    Usage:
        with ChemFigure("fig1_nmr", journal="jacs", width="single") as cf:
            ax = cf.ax
            ax.plot(ppm, intensity)
    """
    name: str
    journal: str = "jacs"
    width: str = "single"         # "single" | "double" | "full"
    height_ratio: float = 0.618   # golden ratio for single-col
    formats: tuple = ("svg", "pdf", "png", "pickle")
    save_data: bool = True

    # Derived
    fig: Optional[plt.Figure] = field(default=None, repr=False)
    ax: Optional[plt.Axes] = field(default=None, repr=False)
    _saved_paths: list = field(default_factory=list, repr=False)

    # Width presets in inches for common journals
    WIDTHS = {
        "single": 3.3,    # single column (JACS / ACS)
        "double": 7.0,    # double column
        "full": 6.0,      # full page
    }

    def __enter__(self):
        use_journal_style(self.journal)
        w = self.WIDTHS.get(self.width, self.WIDTHS["single"])
        h = w * self.height_ratio
        self.fig, self.ax = plt.subplots(figsize=(w, h))
        return self

    def __exit__(self, *args):
        self.fig.tight_layout()
        base = FIGURES_DIR / self.name
        if self.save_data:
            self._save_data(base)
        for fmt in self.formats:
            if fmt == "pickle":
                with open(f"{base}.pickle", "wb") as f:
                    pickle.dump(self.fig, f)
            else:
                self.fig.savefig(f"{base}.{fmt}", dpi=600, format=fmt,
                                 bbox_inches="tight", transparent=False)
            self._saved_paths.append(f"{base}.{fmt}")
        plt.close(self.fig)
        # Report
        print(f"[ChemFigure] {self.name} → {', '.join(f'{fmt}' for fmt in self.formats)}")
        # Update plot log for self-evolution
        _log_plot(self.name, self.journal, self.width, self._saved_paths)

    def _save_data(self, base):
        """Export all line/bar data on the axes to CSV."""
        records = []
        for line in self.ax.get_lines():
            x, y = line.get_xdata(), line.get_ydata()
            label = line.get_label() or "data"
            df = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)})
            csv_path = f"{base}_{label}.csv"
            df.to_csv(csv_path, index=False)
            records.append({"label": label, "csv": csv_path})
        # Also save collections (bar charts, scatter)
        for coll in self.ax.collections:
            offsets = coll.get_offsets()
            if len(offsets) > 0:
                label = coll.get_label() or "scatter"
                df = pd.DataFrame(offsets, columns=["x", "y"])
                csv_path = f"{base}_{label}.csv"
                df.to_csv(csv_path, index=False)
                records.append({"label": label, "csv": csv_path})
        if records:
            with open(f"{base}_sources.json", "w") as f:
                json.dump(records, f, indent=2)


# ---------------------------------------------------------------------------
# Plot experience logger (self-evolution)
# ---------------------------------------------------------------------------

EXPERIENCE_LOG = PROJECT_ROOT / "data" / "plot_experience.jsonl"


def _log_plot(name, journal, width, paths):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "name": name,
        "journal": journal,
        "width": width,
        "paths": {p.split(".")[-1]: p for p in paths},
    }
    with open(EXPERIENCE_LOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_plot_history(n: int = 10) -> list[dict]:
    """Return the last n plot records, for learning from past plots."""
    if not EXPERIENCE_LOG.exists():
        return []
    with open(EXPERIENCE_LOG) as f:
        lines = f.readlines()
    entries = [json.loads(ln) for ln in lines[-n:]]
    return entries


# ---------------------------------------------------------------------------
# Specialized plot functions
# ---------------------------------------------------------------------------

def plot_nmr_1d(x, y, title="", xlabel="δ / ppm", ylabel="",
                highlight_peaks: list = None, color="#1a1a1a",
                invert_x: bool = True, **kwargs):
    """
    Draw a 1D NMR spectrum.
    Chemical convention: x-axis inverted (right to left: 0 → high ppm).
    Peaks face upward.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y, color=color, linewidth=0.8, **kwargs)
    ax.fill_between(x, 0, y, color=color, alpha=0.12)

    if highlight_peaks:
        for px, py in highlight_peaks:
            ax.axvline(px, color="#c0392b", linewidth=0.6, linestyle="--", alpha=0.7)
            ax.annotate(f"{px:.2f}", xy=(px, py), xytext=(0, 8),
                        textcoords="offset points", fontsize=7, color="#c0392b",
                        ha="center")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="normal")
    if invert_x:
        ax.invert_xaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks([])
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig, ax


def plot_mass_spectrum(mz, intensity, title="", mz_range=None,
                       label_peaks: int = 10, color="#1a1a1a"):
    """
    Draw a mass spectrum (stick/centroid plot).
    Automatically labels the top N peaks with their m/z values.
    """
    fig, ax = plt.subplots(figsize=(6, 3))

    # Stick lines
    for m, i in zip(mz, intensity):
        ax.plot([m, m], [0, i], color=color, linewidth=0.6)

    # Label top peaks
    order = np.argsort(intensity)[::-1][:label_peaks]
    for idx in order:
        ax.annotate(f"{mz[idx]:.1f}", xy=(mz[idx], intensity[idx]),
                    xytext=(0, 5), textcoords="offset points",
                    fontsize=6, ha="center", color="#2c3e50", rotation=90)

    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative Intensity")
    ax.set_title(title, fontweight="normal")
    if mz_range:
        ax.set_xlim(mz_range)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks([])
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig, ax


def plot_esr(x, y, title="", xlabel="Magnetic Field / G",
             ylabel="dI/dB", color="#1a1a1a"):
    """Draw an ESR/EPR first-derivative spectrum."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y, color=color, linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="normal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig, ax


def plot_kinetics(time, concentration, labels=None, title="",
                  xlabel="Time / min", ylabel="Concentration / M"):
    """Draw a kinetics trace (concentration vs time)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    time = np.asarray(time)
    concentration = np.asarray(concentration)

    if concentration.ndim == 1:
        concentration = concentration.reshape(-1, 1)

    colors = ["#1a1a1a", "#c0392b", "#2980b9", "#27ae60", "#8e44ad"]
    for i in range(concentration.shape[1]):
        lbl = labels[i] if labels and i < len(labels) else f"Species {i+1}"
        c = colors[i % len(colors)]
        ax.plot(time, concentration[:, i], color=c, linewidth=1.2, label=lbl)
        ax.scatter(time, concentration[:, i], color=c, s=12, edgecolors="white",
                   linewidth=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="normal")
    if labels:
        ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig, ax


def plot_predicted_vs_experimental(pred_x, pred_y, exp_x, exp_y,
                                   title="Predicted vs Experimental",
                                   xlabel=""):
    """
    Overlay predicted (line) and experimental (points) data for comparison.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(pred_x, pred_y, color="#2980b9", linewidth=1.0, label="Predicted")
    ax.scatter(exp_x, exp_y, color="#c0392b", s=16, zorder=5,
               edgecolors="white", linewidth=0.3, label="Experimental")

    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="normal")
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Utility: quick save any matplotlib figure
# ---------------------------------------------------------------------------

def save_figure(fig, name: str, formats=("svg", "pdf", "png")):
    """Save an existing figure in multiple editable formats."""
    saved = []
    for fmt in formats:
        path = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(str(path), dpi=600, format=fmt if fmt != "pickle" else None,
                    bbox_inches="tight")
        saved.append(str(path))
    print(f"[save_figure] {name} → {' '.join(saved)}")
    return saved


print("[plot_utils] Ready — use ChemFigure context manager for editable publication figures.")
