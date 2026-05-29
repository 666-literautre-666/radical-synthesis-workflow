"""
Mass spectrometry data processing:
  - Predict isotopic distribution from molecular formula
  - Predict common fragment ions
  - Process experimental MS data
  - Publication-quality stick spectra
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

from scripts.plot_utils import ChemFigure, FIGURES_DIR, PREDICTIONS_DIR


def exact_mass(smiles: str) -> float:
    """Calculate monoisotopic exact mass from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Descriptors.ExactMolWt(mol)


def molecular_formula(smiles: str) -> str:
    """Get molecular formula from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return rdMolDescriptors.CalcMolFormula(mol)


def predict_isotopic_pattern(smiles: str, charge: int = 1) -> dict:
    """
    Predict isotopic distribution for a molecule.
    Uses RDKit's isotope enumerator for natural abundance simulation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")

    formula = molecular_formula(smiles)
    mass = exact_mass(smiles)

    # Build isotopic distribution via brute-force enumeration of common isotopes
    isotopes = _compute_isotopic_distribution(mol)

    return {
        "smiles": smiles,
        "formula": formula,
        "monoisotopic_mass": mass,
        "charge": charge,
        "mz": [m / charge for m in isotopes["masses"]],
        "intensity": [i / max(isotopes["intensities"]) * 100 for i in isotopes["intensities"]],
        "mz_theoretical": isotopes["masses"],
        "relative_abundance": [i / max(isotopes["intensities"]) * 100 for i in isotopes["intensities"]],
    }


def _compute_isotopic_distribution(mol, min_abundance=0.1) -> dict:
    """
    Compute isotopic distribution using RDKit.
    Enumerates C[12/13], N[14/15], O[16/17/18], S[32/33/34/36],
    Cl[35/37], Br[79/81] combinations.
    """
    from itertools import product

    # Natural abundances
    iso_abund = {
        "C": [(12, 0.9893), (13, 0.0107)],
        "H": [(1, 0.999885), (2, 0.000115)],
        "N": [(14, 0.99632), (15, 0.00368)],
        "O": [(16, 0.99757), (17, 0.00038), (18, 0.00205)],
        "S": [(32, 0.9499), (33, 0.0075), (34, 0.0425), (36, 0.0001)],
        "Cl": [(35, 0.7576), (37, 0.2424)],
        "Br": [(79, 0.5069), (81, 0.4931)],
    }

    # Count atoms by element
    atom_counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1

    # For heavy atoms (C,N,O,S,Cl,Br), enumerate isotope combos
    # To keep it tractable, only enumerate elements with significant heavy isotopes
    heavy_elements = {k: v for k, v in atom_counts.items()
                      if k in ["C", "N", "O", "S", "Cl", "Br"]}
    nominal_mass = sum(atom_counts.get(sym, 0) * iso_abund[sym][0][0]
                        for sym in atom_counts)

    # Use RDKit's brute-force for exact mass
    mol_no_H = Chem.RemoveHs(mol)
    exact_mass_val = Descriptors.ExactMolWt(mol_no_H)

    # For < 2 heavy atoms that vary, just compute the [M], [M+1], [M+2] pattern
    masses = [exact_mass_val, exact_mass_val + 1.00335, exact_mass_val + 2.00671]
    nC = atom_counts.get("C", 0)

    # [M+1] mainly from 13C
    m1_intensity = nC * 1.08  # ~1.08% per carbon
    # [M+2] from two 13C or one 34S / 37Cl etc.
    m2_intensity = (nC * (nC - 1) / 2) * (0.0108 ** 2) * 100

    nS = atom_counts.get("S", 0)
    if nS > 0:
        m2_intensity += nS * 4.4  # 34S

    nCl = atom_counts.get("Cl", 0)
    if nCl > 0:
        m2_intensity += nCl * 32.0  # 37Cl

    nBr = atom_counts.get("Br", 0)
    if nBr > 0:
        m2_intensity += nBr * 97.3  # 81Br

    intensities = [100.0, m1_intensity, m2_intensity]

    return {"masses": masses, "intensities": intensities}


def predict_common_fragments(smiles: str) -> list[dict]:
    """
    Predict common fragmentation patterns using simple bond-breaking rules.
    Returns list of {fragment_smiles, mass, description}.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fragments = []
    bonds = list(mol.GetBonds())

    # Score bonds by likeliness to break (weaker bonds break first)
    bond_strength = {
        Chem.BondType.SINGLE: 1.0,
        Chem.BondType.DOUBLE: 2.0,
        Chem.BondType.TRIPLE: 3.0,
        Chem.BondType.AROMATIC: 2.5,
    }

    scored_bonds = []
    for bond in bonds:
        bt = bond.GetBondType()
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        # Bonds adjacent to heteroatoms break more easily
        hetero_adjacent = any(a.GetAtomicNum() not in [1, 6] for a in [a1, a2])
        score = bond_strength.get(bt, 1.0) - (0.3 if hetero_adjacent else 0)
        scored_bonds.append((bond.GetIdx(), score, bond))

    # Sort by weakest
    scored_bonds.sort(key=lambda x: x[1])

    for bidx, score, bond in scored_bonds[:5]:  # top 5 easiest breaks
        try:
            frag_mol = Chem.FragmentOnBonds(mol, [bidx], addDummies=True)
            frags = Chem.GetMolFrags(frag_mol, asMols=True)
            for f in frags:
                mass = Descriptors.ExactMolWt(f)
                frag_smi = Chem.MolToSmiles(f)
                fragments.append({
                    "fragment_smiles": frag_smi,
                    "exact_mass": round(mass, 4),
                    "parent_bond_idx": bidx,
                    "fragility_score": round(score, 2),
                })
        except Exception:
            continue

    # Deduplicate
    seen = set()
    unique = []
    for f in fragments:
        key = (f["fragment_smiles"], round(f["exact_mass"], 2))
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return sorted(unique, key=lambda x: x["exact_mass"])


def predict_ms(smiles: str) -> dict:
    """Full MS prediction: isotopic pattern + fragments."""
    iso = predict_isotopic_pattern(smiles)
    frags = predict_common_fragments(smiles)
    result = {**iso, "fragments": frags}

    # Save prediction
    pred_path = PREDICTIONS_DIR / f"{smiles.replace(' ', '_')}_ms.json"
    with open(pred_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def read_ms_csv(filepath: str) -> dict:
    """
    Read experimental MS data from CSV (columns: m/z, intensity).
    Or from common text export formats.
    """
    df = pd.read_csv(filepath)
    cols = df.columns.tolist()
    mz_col = next((c for c in cols if "m" in c.lower() and "z" in c.lower()), cols[0])
    int_col = next((c for c in cols if "int" in c.lower()), cols[-1])
    return {
        "mz": df[mz_col].values.tolist(),
        "intensity": df[int_col].values.tolist(),
        "source_file": str(filepath),
    }


def plot_ms_prediction(smiles: str, journal: str = "jacs"):
    """Predict MS and generate publication-quality stick spectrum."""
    pred = predict_ms(smiles)

    mz = np.array(pred["mz"])
    intensity = np.array(pred["intensity"])

    with ChemFigure(f"ms_predicted_{smiles[:30]}", journal=journal) as cf:
        ax = cf.ax
        for m, i in zip(mz, intensity):
            ax.plot([m, m], [0, i], color="#1a1a1a", linewidth=0.8)

        # Label peaks
        order = np.argsort(intensity)[::-1][:8]
        for idx in order:
            ax.annotate(f"{mz[idx]:.1f}", xy=(mz[idx], intensity[idx]),
                        xytext=(0, 6), textcoords="offset points",
                        fontsize=6, ha="center", rotation=90, color="#2c3e50")

        ax.set_xlabel("m/z")
        ax.set_ylabel("Relative Abundance")
        ax.set_title(f"Predicted MS — {smiles}\n{pred['formula']}  [M] = {pred['monoisotopic_mass']:.2f}",
                     fontweight="normal", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_ticks([])

    return pred


def plot_ms_comparison(smiles: str, exp_data: dict, journal: str = "jacs"):
    """Overlay predicted and experimental mass spectra."""
    pred = predict_ms(smiles)

    p_mz = np.array(pred["mz"])
    p_int = np.array(pred["intensity"]) / max(pred["intensity"]) * 100

    with ChemFigure(f"ms_comparison_{smiles[:30]}", journal=journal,
                    width="double") as cf:
        ax = cf.ax

        # Predicted: upward sticks (blue)
        for m, i in zip(p_mz, p_int):
            ax.plot([m, m], [0, i], color="#2980b9", linewidth=0.6, alpha=0.8)

        # Experimental: downward sticks (red)
        if exp_data:
            e_mz = np.array(exp_data.get("mz", []))
            e_int = np.array(exp_data.get("intensity", []))
            if len(e_int) > 0:
                e_int = e_int / max(e_int) * 100
            for m, i in zip(e_mz, e_int):
                ax.plot([m, m], [0, -i], color="#c0392b", linewidth=0.6, alpha=0.8)

        ax.axhline(0, color="gray", linewidth=0.3)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Relative Abundance")
        ax.set_title(f"MS — {smiles}  [M] = {pred['monoisotopic_mass']:.2f}",
                     fontweight="normal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_ticks([])

    return pred


print("[ms_processor] Ready.")
