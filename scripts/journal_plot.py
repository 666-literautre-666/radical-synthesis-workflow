"""
Journal-quality chemistry plotting engine.
Features:
  - RDKit molecular structure inset on every figure
  - NMR multiplet simulation with J-coupling
  - Integration curve overlay
  - Multi-region zoom panels
  - Self-evolution feedback tracking
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from pathlib import Path
from io import BytesIO
import pickle, json, datetime

# ---------------------------------------------------------------------------
# Molecule drawing (RDKit → matplotlib inset)
# ---------------------------------------------------------------------------

def draw_molecule_inset(ax, smiles, position="upper right", scale=0.22):
    """
    Draw a molecular structure as an inset on the given matplotlib axes.
    Uses RDKit to render a high-quality 2D structure.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return

    AllChem.Compute2DCoords(mol)

    # Render to SVG-like coordinates via RDKit's high-quality drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.bondLineWidth = 2.5
    opts.atomLabelFontSize = 28
    opts.fontSize = 10
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_data = drawer.GetDrawingText()

    # Load as matplotlib image
    from matplotlib.offsetbox import OffsetImage, AnchoredOffsetbox
    import matplotlib.image as mpimg
    from io import BytesIO

    img = mpimg.imread(BytesIO(png_data), format='png')

    # Position the inset
    bbox_anchor = {
        "upper right": (1.02, 1.0),
        "upper left": (-0.02, 1.0),
        "lower right": (1.02, 0.0),
        "lower left": (-0.02, 0.0),
    }.get(position, (1.02, 1.0))

    imagebox = OffsetImage(img, zoom=scale, interpolation="lanczos")
    ab = AnchoredOffsetbox(loc=position.replace(" ", " "),
                           child=imagebox, pad=0, frameon=True,
                           bbox_to_anchor=bbox_anchor,
                           bbox_transform=ax.transAxes, borderpad=0)
    ax.add_artist(ab)


# ---------------------------------------------------------------------------
# NMR Multiplet simulation (J-coupling based)
# ---------------------------------------------------------------------------

# Typical vicinal coupling constants (Hz) by environment
# Reference: Pretsch, "Structure Determination of Organic Compounds"
J_COUPLING_TABLE = {
    # Aliphatic vicinal (3J_HH)
    "alkane_free_rotation": (6, 8),     # freely rotating CH2-CH2
    "alkane_CH_CH2": (5, 7),            # CH-CH2
    "alkane_CH_CH": (5, 8),             # CH-CH
    # Substituted ethane fragments
    "alpha_oxygen_CH2": (5, 7),         # -O-CH2-CH3
    "alpha_carbonyl_CH2": (6, 8),       # -CO-CH2-CH3
    "alpha_aromatic_CH2": (7, 8),       # Ar-CH2-CH3
    # Alkene
    "alkene_cis": (6, 12),
    "alkene_trans": (12, 18),
    "alkene_geminal": (0, 3),
    # Aromatic
    "aromatic_ortho": (7, 9),
    "aromatic_meta": (1, 3),
    "aromatic_para": (0, 1),
    # Geminal (2J_HH)
    "geminal_alkane": (-12, -15),
    "geminal_alpha_O": (-9, -12),
    # Long range
    "allylic": (0, 3),
    "homoallylic": (0, 2),
    "w_coupling": (0, 2),
}


def simulate_multiplet(center_ppm, j_values, j_multiplicities, spectrometer_freq=400,
                       peak_width=0.3, n_points=256):
    """
    Simulate a first-order multiplet pattern.

    Parameters:
      center_ppm: chemical shift center
      j_values: list of coupling constants in Hz
      j_multiplicities: list of multiplicities (2=doublet, 3=triplet, 4=quartet, etc.)
      spectrometer_freq: MHz (for converting Hz → ppm)
      peak_width: Lorentzian half-width in Hz
      n_points: points per multiplet

    Returns:
      x_ppm, y (normalized multiplet lineshape)
    """
    # Generate stick spectrum
    lines = [0.0]
    for j, mult in zip(j_values, j_multiplicities):
        new_lines = []
        for pos in lines:
            for k in range(mult):
                offset = (k - (mult - 1) / 2) * j
                new_lines.append(pos + offset)
        lines = new_lines

    # Sort by position
    lines = np.array(sorted(lines))

    # Convert Hz to ppm and build Lorentzian spectrum
    half_range = (max(j_values) * max(j_multiplicities) * 1.5) / spectrometer_freq if j_values else 0.02
    half_range = max(half_range, 0.01)

    x = np.linspace(center_ppm - half_range, center_ppm + half_range, n_points)
    y = np.zeros(n_points)

    pw_ppm = peak_width / spectrometer_freq
    for pos in lines:
        y += 1.0 / (1.0 + ((x - (center_ppm + pos / spectrometer_freq)) / pw_ppm) ** 2)

    y = y / np.max(y) if np.max(y) > 0 else y
    return x, y


def predict_multiplets_from_smiles(smiles):
    """
    Predict 1H NMR chemical shifts and coupling patterns from SMILES.
    Returns list of (center_ppm, total_H, coupling_list, multiplicity_list, label).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)

    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass

    signals = []
    # Map carbon idx → proton chemical shift range
    shift_ranges = {
        "alkane_CH3": (0.7, 1.3), "alkane_CH2": (1.1, 1.5),
        "alkane_CH": (1.4, 1.8), "alpha_to_carbonyl": (2.0, 2.6),
        "alpha_to_oxygen": (3.3, 4.0), "alpha_to_nitrogen": (2.5, 3.2),
        "alkene": (4.5, 6.5), "aromatic": (6.5, 8.5),
        "aldehyde": (9.5, 10.5), "alpha_to_halogen": (3.0, 4.5),
    }

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue

        neighbors = [n for n in atom.GetNeighbors()]
        nH = sum(1 for n in neighbors if n.GetAtomicNum() == 1)
        if nH == 0:
            continue

        # Classify environment
        neighbor_syms = [n.GetSymbol() for n in neighbors]
        is_aromatic = atom.GetIsAromatic()
        env = "alkane_CH" + str(nH)

        if is_aromatic:
            env = "aromatic"
        elif "O" in neighbor_syms:
            env = "alpha_to_oxygen"
        elif "N" in neighbor_syms:
            env = "alpha_to_nitrogen"
        elif any(s in ["F", "Cl", "Br", "I"] for s in neighbor_syms):
            env = "alpha_to_halogen"
        elif any(n.GetAtomicNum() == 6 and
                 mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                 for n in neighbors):
            env = "alkene"

        rng = shift_ranges.get(env, (1.0, 3.0))
        center = (rng[0] + rng[1]) / 2 + np.random.normal(0, 0.08)

        # Estimate coupling: count vicinal protons on adjacent carbons
        couplings = []
        multiplicities = []
        for nbr in neighbors:
            if nbr.GetAtomicNum() == 6:  # adjacent carbon
                nH_vic = sum(1 for nn in nbr.GetNeighbors() if nn.GetAtomicNum() == 1)
                if nH_vic > 0:
                    j = np.random.uniform(6, 8)  # typical 3J ~7 Hz
                    couplings.append(j)
                    multiplicities.append(nH_vic + 1)  # n+1 rule

        if not couplings:
            couplings = [0]
            multiplicities = [1]

        signals.append({
            "center_ppm": center,
            "n_protons": nH,
            "couplings_Hz": couplings,
            "multiplicities": multiplicities,
            "environment": env,
        })

    return signals


# ---------------------------------------------------------------------------
# Integration curve
# ---------------------------------------------------------------------------

def compute_integration_curve(x_ppm, y_spectrum):
    """Compute cumulative integration curve for display on NMR plots."""
    integral = np.cumsum(y_spectrum)
    integral = integral / np.max(np.abs(integral)) * 0.4  # scale to 40% of plot
    return integral


# ---------------------------------------------------------------------------
# Self-evolution: plot feedback tracker
# ---------------------------------------------------------------------------

class PlotMemory:
    """Tracks plotting decisions and user corrections to improve over time."""

    def __init__(self, storage_path: Path):
        self.path = storage_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = self._load()

    def _load(self):
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(l) for l in f.readlines() if l.strip()]

    def record(self, plot_type, params, corrections=None):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "plot_type": plot_type,
            "params": params,
            "corrections": corrections or {},
        }
        self.entries.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_best_params(self, plot_type):
        """Return the most recently corrected params for a given plot type."""
        relevant = [e for e in self.entries if e["plot_type"] == plot_type and e["corrections"]]
        if relevant:
            return relevant[-1]["corrections"]
        return {}

    def stats(self):
        return {
            "total": len(self.entries),
            "by_type": {},
            "corrected": sum(1 for e in self.entries if e.get("corrections")),
        }


# Global instance
_memory = None

def get_plot_memory():
    global _memory
    if _memory is None:
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent / "data" / "plot_memory.jsonl"
        _memory = PlotMemory(p)
    return _memory


# ---------------------------------------------------------------------------
# Multi-panel NMR figure (aromatic + aliphatic regions)
# ---------------------------------------------------------------------------

def nmr_multipanel(ppm, spectrum, regions=None, smiles="", title=""):
    """
    Create a multi-panel NMR figure with:
      - Full spectrum (top)
      - Aromatic zoom (6.5-8.5 ppm, bottom left)
      - Aliphatic zoom (0.5-4.5 ppm, bottom right)
      - Molecule structure inset
      - Integration curve
    """
    if regions is None:
        regions = [
            ("full", 10, -1),
            ("aromatic", 8.5, 6.0),
            ("aliphatic", 4.5, 0.5),
        ]

    n_regions = len(regions)
    fig = plt.figure(figsize=(7, 2.5 * n_regions))
    gs = fig.add_gridspec(n_regions, 2, width_ratios=[4, 1],
                          hspace=0.35, wspace=0.05)

    axes = {}
    for i, (name, x_l, x_r) in enumerate(regions):
        ax = fig.add_subplot(gs[i, 0])
        axes[name] = ax
        mask = (ppm <= x_l) & (ppm >= x_r)
        ax.plot(ppm[mask], spectrum[mask], color="#1a1a1a", linewidth=0.7)
        ax.fill_between(ppm[mask], 0, spectrum[mask], color="#1a1a1a", alpha=0.08)
        ax.invert_xaxis()
        ax.set_xlim(x_l, x_r)
        ax.yaxis.set_ticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        if i < n_regions - 1:
            ax.set_xticklabels([])
        if i == 0:
            ax.set_title(title, fontsize=9, fontweight="normal")

    axes[list(axes.keys())[-1]].set_xlabel(r"$\delta$ / ppm", fontsize=8)

    # Molecule drawing in right column
    ax_mol = fig.add_subplot(gs[:, 1])
    ax_mol.axis("off")
    if smiles:
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw, AllChem
            from rdkit.Chem.Draw import rdMolDraw2D
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox

            mol = Chem.MolFromSmiles(smiles)
            if mol:
                AllChem.Compute2DCoords(mol)
                drawer = rdMolDraw2D.MolDraw2DCairo(300, 400)
                opts = drawer.drawOptions()
                opts.bondLineWidth = 3
                opts.atomLabelFontSize = 32
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                import matplotlib.image as mpimg
                img = mpimg.imread(BytesIO(drawer.GetDrawingText()), format='png')
                ax_mol.imshow(img, aspect="equal")
        except Exception:
            ax_mol.text(0.5, 0.5, smiles, transform=ax_mol.transAxes,
                        ha="center", va="center", fontsize=9, color="gray")

    return fig, axes


# ---------------------------------------------------------------------------
# High-quality figure saver
# ---------------------------------------------------------------------------

def save_journal_figure(fig, path, name):
    """Save a figure in all publishable formats with metadata."""
    base = Path(path) / name
    formats = {
        "svg": {"dpi": 600, "format": "svg"},
        "pdf": {"dpi": 600, "format": "pdf"},
        "png": {"dpi": 600, "format": "png"},
    }
    saved = []
    for fmt, kwargs in formats.items():
        fpath = f"{base}.{fmt}"
        fig.savefig(fpath, bbox_inches="tight", **kwargs)
        saved.append(fpath)

    # Pickle for re-editing
    with open(f"{base}.pickle", "wb") as f:
        pickle.dump(fig, f)

    # Log for self-evolution
    mem = get_plot_memory()
    mem.record("figure_save", {"name": name, "formats": list(formats.keys())})

    return saved


print("[journal_plot] Enhanced plotting engine ready.")
print("  - molecule structure insets")
print("  - NMR multiplet simulation")
print("  - integration curves")
print("  - multi-panel figures")
print("  - self-evolution memory")
