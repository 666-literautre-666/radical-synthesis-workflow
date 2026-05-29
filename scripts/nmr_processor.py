"""
NMR data processing:
  - Predict 1H/13C chemical shifts from SMILES (RDKit-based)
  - Read & process experimental Bruker / Varian / JEOL data (via nmrglue)
  - Peak picking, integration, multiplet analysis
  - Predicted vs experimental overlay
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from scripts.plot_utils import ChemFigure, FIGURES_DIR, PREDICTIONS_DIR, EXPERIMENTAL_DIR

# ---------------------------------------------------------------------------
# Chemical shift prediction (additive model)
# ---------------------------------------------------------------------------

# Approximate 1H shift ranges by chemical environment (ppm)
# Reference: Pretsch, "Structure Determination of Organic Compounds"
PROTON_SHIFT_RANGES = {
    "alkane_CH3": (0.7, 1.3),
    "alkane_CH2": (1.1, 1.5),
    "alkane_CH": (1.4, 1.8),
    "alpha_to_carbonyl": (2.0, 2.6),
    "allylic": (1.6, 2.2),
    "benzylic": (2.2, 3.0),
    "alpha_to_oxygen": (3.3, 4.0),
    "alpha_to_nitrogen": (2.5, 3.2),
    "alkyne": (2.0, 3.0),
    "alkene": (4.5, 6.5),
    "aromatic": (6.5, 8.5),
    "aldehyde": (9.5, 10.5),
    "carboxylic_acid": (10.0, 13.0),
    "alcohol": (1.0, 5.5),
    "amine": (1.0, 3.0),
    "amide": (5.0, 8.0),
    "alpha_to_halogen": (3.0, 4.5),
}

# Approximate 13C shift ranges
CARBON_SHIFT_RANGES = {
    "alkane": (0, 50),
    "alpha_to_carbonyl": (25, 50),
    "alkene": (100, 150),
    "aromatic": (110, 160),
    "carbonyl_ester_amide": (155, 180),
    "carbonyl_ketone": (190, 220),
    "carbonyl_aldehyde": (190, 210),
    "alkyne": (70, 90),
    "nitrile": (115, 125),
    "C-O": (60, 80),
    "C-N": (40, 65),
    "C-X": (30, 70),
}


def _classify_proton_env(mol, atom_idx):
    """Simple classification of proton chemical environment."""
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetAtomicNum() != 6:
        return None

    neighbors = [n for n in atom.GetNeighbors()]
    nH = sum(1 for n in neighbors if n.GetAtomicNum() == 1)
    if nH == 0:
        return None

    # Check neighbor types
    neighbor_syms = [n.GetSymbol() for n in neighbors]
    is_aromatic = atom.GetIsAromatic()
    bond_types = [mol.GetBondBetweenAtoms(atom_idx, n.GetIdx()).GetBondType() for n in neighbors]

    if is_aromatic:
        return "aromatic"
    if any(bt == Chem.BondType.DOUBLE for bt in bond_types):
        return "alkene"
    if any(bt == Chem.BondType.TRIPLE for bt in bond_types):
        return "alkyne"
    if "O" in neighbor_syms:
        return "alpha_to_oxygen"
    if "N" in neighbor_syms:
        return "alpha_to_nitrogen"
    if any(s in ["F", "Cl", "Br", "I"] for s in neighbor_syms):
        return "alpha_to_halogen"

    # Check beta to C=O
    for nbr in neighbors:
        for nbr2 in nbr.GetNeighbors():
            if nbr2.GetIdx() != atom_idx:
                b = mol.GetBondBetweenAtoms(nbr.GetIdx(), nbr2.GetIdx())
                if b and b.GetBondType() == Chem.BondType.DOUBLE and nbr2.GetAtomicNum() == 8:
                    return "alpha_to_carbonyl"

    # Simple alkane
    if nH == 3:
        return "alkane_CH3"
    elif nH == 2:
        return "alkane_CH2"
    else:
        return "alkane_CH"


def predict_nmr_from_smiles(smiles: str, n_conformers: int = 50) -> dict:
    """
    Predict 1H and 13C chemical shifts from SMILES.
    Returns a dict with predicted shifts and a rough simulated spectrum.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    proton_shifts = []
    carbon_shifts = []
    atom_labels = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        sym = atom.GetSymbol()
        env = None

        if sym == "C":
            # Count explicit H neighbors (after AddHs, Hs are explicit)
            nH = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
            env = _classify_proton_env(mol, idx)
            range_tup = PROTON_SHIFT_RANGES.get(env) if env else None
            if nH > 0 and range_tup:
                # Use midpoint of range + small random perturbation
                shift = np.mean(range_tup) + np.random.normal(0, 0.1)
                for _ in range(nH):
                    proton_shifts.append(shift)
                    atom_labels.append(f"C{idx}-{env or 'unknown'}")

        # 13C estimation (crude — for a real project, use DFT or ML)
        if sym == "C" and not atom.GetIsAromatic():
            nH_local = atom.GetTotalNumHs()
            is_carbonyl = any(
                mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE and n.GetAtomicNum() == 8
                for n in atom.GetNeighbors()
            )
            if is_carbonyl:
                carbon_shifts.append(200 + np.random.normal(0, 5))
                atom_labels.append(f"C{idx}-carbonyl")
            else:
                base = 10 + nH_local * 10
                carbon_shifts.append(base + np.random.normal(0, 3))
                atom_labels.append(f"C{idx}-alkane")

    proton_shifts = np.array(proton_shifts)
    carbon_shifts = np.array(carbon_shifts)

    # Build simulated 1H spectrum
    h_x, h_y = _build_spectrum(proton_shifts, x_range=(10, 0), peak_width=0.02)
    c_x, c_y = _build_spectrum(carbon_shifts, x_range=(220, 0), peak_width=1.0)

    return {
        "smiles": smiles,
        "proton_shifts": proton_shifts.tolist(),
        "proton_atom_labels": [l for l in atom_labels if "C" in l][:len(proton_shifts)],
        "predicted_h_spectrum": {"x": h_x.tolist(), "y": h_y.tolist()},
        "predicted_c_spectrum": {"x": c_x.tolist(), "y": c_y.tolist()},
        "carbon_shifts": carbon_shifts.tolist(),
    }


def _build_spectrum(peak_positions, x_range, peak_width, n_points=4096):
    """Build a Lorentzian-convolved spectrum from peak positions."""
    if len(peak_positions) == 0:
        x = np.linspace(x_range[0], x_range[1], n_points)
        return x, np.zeros(n_points)
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.zeros(n_points)
    for pos in peak_positions:
        y += 1.0 / (1.0 + ((x - pos) / peak_width) ** 2)
    return x, y


# ---------------------------------------------------------------------------
# Experimental data processing (nmrglue)
# ---------------------------------------------------------------------------

def read_bruker_nmr(data_dir: str) -> dict:
    """
    Read Bruker format NMR data from a directory (contains fid, acqu, etc.).
    Returns dict with time-domain data and acquisition parameters.
    """
    import nmrglue as ng

    dic, data = ng.bruker.read(data_dir)
    # Fourier transform
    udic = ng.bruker.guess_udic(dic, data)
    data_ft = ng.proc_base.fft(data)
    # Phase correction (simple zero-order)
    data_ft = ng.proc_base.ps(data_ft, p0=0.0)
    data_ft = ng.proc_base.di(data_ft)

    # Get ppm scale
    udic[0]["size"] = data_ft.shape[-1]
    uc = ng.fileiobase.uc_from_udic(udic)
    ppm = uc.ppm_scale()

    return {
        "ppm": ppm.tolist(),
        "spectrum_real": data_ft.real.tolist(),
        "spectrum_imag": data_ft.imag.tolist(),
        "acquisition_params": {k: str(v) for k, v in dic.items() if "acq" in k.lower()},
    }


def read_jeol_nmr(filepath: str) -> dict:
    """Read JEOL-format NMR data."""
    import nmrglue as ng
    dic, data = ng.jeol.read(filepath)
    data_ft = ng.proc_base.fft(data)
    udic = ng.jeol.guess_udic(dic, data)
    udic[0]["size"] = data_ft.shape[-1]
    uc = ng.fileiobase.uc_from_udic(udic)
    return {"ppm": uc.ppm_scale().tolist(), "spectrum": data_ft.real.tolist()}


def peek_pick(ppm, spectrum, height_frac=0.05, min_distance=0.01) -> pd.DataFrame:
    """
    Simple peak picking: find local maxima above a height threshold.
    Returns DataFrame with ppm, height, relative integration.
    """
    from scipy.signal import find_peaks

    ppm = np.asarray(ppm)
    spectrum = np.asarray(spectrum)
    height = np.max(spectrum) * height_frac
    min_dist = int(len(ppm) * min_distance / (ppm[0] - ppm[-1]))

    peaks, props = find_peaks(spectrum, height=height, distance=max(1, min_dist))

    records = []
    for i, idx in enumerate(peaks):
        left = max(idx - 20, 0)
        right = min(idx + 20, len(spectrum))
        area = np.trapezoid(spectrum[left:right], ppm[left:right])
        records.append({
            "peak_index": i + 1,
            "ppm": ppm[idx],
            "intensity": spectrum[idx],
            "integration_area": abs(area),
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        total = df["integration_area"].sum()
        df["relative_integration"] = df["integration_area"] / total if total > 0 else 0
        df["proton_count_estimate"] = np.round(df["relative_integration"] * df.shape[0], 1)

    return df


# ---------------------------------------------------------------------------
# High-level workflow: predict + compare
# ---------------------------------------------------------------------------

def predict_and_compare_nmr(smiles: str, exp_data: dict = None,
                            journal: str = "jacs") -> dict:
    """
    Main workflow:
    1. Predict NMR from SMILES
    2. If experimental data provided, overlay comparison
    3. Save publication-quality figure
    """
    pred = predict_nmr_from_smiles(smiles)

    # Save prediction
    pred_path = PREDICTIONS_DIR / f"{smiles.replace(' ', '_')}_nmr.json"
    with open(pred_path, "w") as f:
        json.dump(pred, f, indent=2, default=str)

    # Plot predicted 1H spectrum
    px = np.array(pred["predicted_h_spectrum"]["x"])
    py = np.array(pred["predicted_h_spectrum"]["y"])

    with ChemFigure(f"nmr_predicted_{smiles[:20]}", journal=journal) as cf:
        ax = cf.ax
        ax.plot(px, py, color="#1a1a1a", linewidth=0.8)
        ax.fill_between(px, 0, py, color="#1a1a1a", alpha=0.1)
        ax.set_xlabel("δ / ppm")
        ax.set_ylabel("")
        ax.set_title(f"Predicted $^1$H NMR — {smiles}", fontweight="normal")
        ax.invert_xaxis()
        ax.yaxis.set_ticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # If experimental data, make comparison
    if exp_data:
        exp_x = np.array(exp_data.get("ppm", []))
        exp_y = np.array(exp_data.get("spectrum_real", exp_data.get("spectrum", [])))

        with ChemFigure(f"nmr_comparison_{smiles[:20]}", journal=journal,
                        width="double") as cf:
            ax = cf.ax
            ax.plot(px, py / np.max(py) if np.max(py) > 0 else 1,
                    color="#2980b9", linewidth=0.8, label="Predicted")
            if len(exp_y) > 0:
                ax.plot(exp_x, exp_y / np.max(exp_y) if np.max(exp_y) > 0 else 1,
                        color="#c0392b", linewidth=0.6, label="Experimental")
            ax.set_xlabel("δ / ppm")
            ax.set_title(f"$^1$H NMR — Predicted vs Experimental", fontweight="normal")
            ax.invert_xaxis()
            ax.yaxis.set_ticks([])
            ax.legend(fontsize=7, frameon=False)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)

    return pred


print("[nmr_processor] Ready.")
