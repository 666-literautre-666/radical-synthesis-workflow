"""
ESR/EPR data processing for radical chemistry:
  - Predict g-values for common radical types
  - Simulate hyperfine coupling patterns
  - Process experimental ESR spectra
  - g-value calculation from frequency/field
  - Spin quantification (double integration vs reference)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json

from scripts.plot_utils import ChemFigure, FIGURES_DIR, EXPERIMENTAL_DIR

# Planck constant h (J·s), Bohr magneton mu_B (J/T)
H = 6.62607015e-34
MU_B = 9.274009994e-24
BETA_E = MU_B / H  # 1.3996 MHz/G (electron gyromagnetic ratio in frequency/field)


# ---------------------------------------------------------------------------
# g-value reference data for common radicals
# ---------------------------------------------------------------------------

RADICAL_G_VALUES = {
    # Organic radicals
    "alkyl_radical": (2.0025, 2.0030, "Alkyl radical R."),
    "benzyl_radical": (2.0025, 2.0030, "Benzyl-type radical"),
    "allyl_radical": (2.0025, 2.0035, "Allyl radical"),
    "phenoxyl_radical": (2.0040, 2.0055, "Phenoxyl radical ArO."),
    "nitroxide": (2.0050, 2.0070, "Nitroxide radical (TEMPO etc.)"),
    "nitroxyl_radical": (2.0050, 2.0065, "Nitroxyl radical >N-O."),
    "semiquinone": (2.0040, 2.0050, "Semiquinone radical anion"),
    "ketyl_radical": (2.0030, 2.0040, "Ketyl radical >C.-O⁻"),
    "aryl_radical": (2.0000, 2.0020, "Aryl σ radical"),
    "peroxy_radical": (2.0100, 2.0400, "Peroxyl radical ROO."),
    "superoxide": (2.0100, 2.0200, "Superoxide O2.⁻"),
    "sulfur_radical": (2.0100, 2.0300, "Thiyl radical RS."),
    "iminyl_radical": (2.0030, 2.0050, "Iminyl radical >C=N."),
    "aminyl_radical": (2.0030, 2.0050, "Aminyl radical >N."),
    # Carbon-centered radicals with heteroatom substitution
    "alpha_alkoxy": (2.0030, 2.0040, "α-alkoxy carbon radical"),
    "alpha_amino": (2.0030, 2.0045, "α-amino carbon radical"),
    "acyl_radical": (2.0000, 2.0020, "Acyl radical R-CO."),
    # Transition metal complexes
    "Cu2+_complex": (2.0500, 2.4000, "Cu(II) complex (axial)"),
    "Fe3+_high_spin": (2.0000, 2.1000, "Fe(III) high-spin"),
    "Mn2+": (1.9800, 2.0200, "Mn(II) high-spin"),
    "VO2+": (1.9300, 1.9900, "Vanadyl VO²⁺"),
}


def predict_g_value(radical_type: str) -> dict:
    """Predict g-value range for a given radical type."""
    info = RADICAL_G_VALUES.get(radical_type.lower())
    if not info:
        g_min, g_max = 2.0000, 2.0100
        description = f"Unknown radical type: {radical_type}"
    else:
        g_min, g_max, description = info

    g_center = (g_min + g_max) / 2
    return {
        "radical_type": radical_type,
        "description": description,
        "g_iso_predicted": round(g_center, 5),
        "g_range": [round(g_min, 5), round(g_max, 5)],
    }


def estimate_g_from_freq_field(mw_freq_ghz: float, resonance_field_g: float) -> float:
    """
    Calculate g-value from microwave frequency and resonance field.
    g = h * nu / (mu_B * B)
    For nu in GHz, B in Gauss: g = 714.477 * nu_GHz / B_G
    """
    if resonance_field_g <= 0:
        return 0.0
    g = 714.477 * mw_freq_ghz / resonance_field_g
    return round(g, 5)


def predict_hyperfine_pattern(radical_smiles: str = None,
                              nuclei: list[dict] = None) -> dict:
    """
    Predict hyperfine splitting pattern.
    nuclei: list of dicts like [{"nucleus": "14N", "coupling_G": 14.5, "multiplicity": 3}, ...]

    Multiplicity = 2*I + 1 for nuclear spin I:
      1H: I=1/2 → doublet (2)
      14N: I=1 → triplet (3)
      2H: I=1 → triplet (3)
      13C: I=1/2 → doublet (2)
      19F: I=1/2 → doublet (2)
      31P: I=1/2 → doublet (2)
    """
    if not nuclei:
        # Default: simple alkyl radical, one alpha-H coupling
        nuclei = [
            {"nucleus": "1H", "coupling_G": 22.0, "multiplicity": 2, "n": 2},
        ]

    # Build splitting pattern
    all_lines = [0.0]  # center
    all_intensities = [1.0]

    for nuc in nuclei:
        n = nuc.get("n", 1)
        a = nuc["coupling_G"]

        for _ in range(n):
            new_lines = []
            new_int = []
            mult = nuc["multiplicity"]

            for center, amp in zip(all_lines, all_intensities):
                # Equal spacing for first-order
                for i in range(mult):
                    offset = (i - (mult - 1) / 2) * a
                    new_lines.append(center + offset)
                    # Pascal's triangle intensities for I=1/2
                    if mult == 2:
                        new_int.append(amp * 1)
                    elif mult == 3:
                        # 1:1:1 for I=1
                        new_int.append(amp * 1)
                    else:
                        new_int.append(amp * 1.0 / mult)

            all_lines = new_lines
            all_intensities = new_int

    # Sort by position
    order = np.argsort(all_lines)
    sorted_lines = np.array(all_lines)[order]
    sorted_intensities = np.array(all_intensities)[order]

    return {
        "line_positions_G": sorted_lines.tolist(),
        "relative_intensities": sorted_intensities.tolist(),
        "nuclei": nuclei,
        "total_lines": len(sorted_lines),
    }


# ---------------------------------------------------------------------------
# Experimental ESR processing
# ---------------------------------------------------------------------------

def read_esr_txt(filepath: str, x_col: int = 0, y_col: int = 1,
                 skip_rows: int = 0) -> dict:
    """Read ESR data from a simple text/CSV file."""
    data = np.loadtxt(filepath, skiprows=skip_rows)
    return {
        "field_G": data[:, x_col].tolist(),
        "signal": data[:, y_col].tolist(),
        "source": str(filepath),
    }


def calculate_g_values(exp_data: dict, mw_freq_ghz: float = 9.8) -> dict:
    """
    From experimental ESR data, find g-values by locating zero-crossings
    and peaks in the derivative spectrum.
    Assumes first-derivative lineshape (standard cw-ESR).
    """
    field = np.array(exp_data.get("field_G", []))
    signal = np.array(exp_data.get("signal", []))

    if len(field) == 0:
        return {"error": "No data"}

    # Zero crossings (inflection points) in 1st derivative → center of resonance
    zero_crossings = []
    for i in range(1, len(signal)):
        if signal[i-1] * signal[i] <= 0 and abs(signal[i] - signal[i-1]) > 0:
            # Linear interpolation
            f_interp = field[i-1] + (0 - signal[i-1]) * (field[i] - field[i-1]) / (signal[i] - signal[i-1])
            zero_crossings.append(f_interp)

    # Peak-to-peak linewidth
    peak_idx = np.argmax(signal)
    trough_idx = np.argmin(signal)
    dHpp = abs(field[peak_idx] - field[trough_idx])

    g_values = [estimate_g_from_freq_field(mw_freq_ghz, b) for b in zero_crossings]

    # Spin count (double integration compared to reference)
    first_integral = np.cumsum(signal)
    # Correct baseline drift for 1st integral
    first_integral = first_integral - np.polyval(np.polyfit(field, first_integral, 1), field)
    second_integral = np.trapezoid(first_integral, field)

    return {
        "microwave_freq_ghz": mw_freq_ghz,
        "zero_crossing_fields_G": zero_crossings,
        "g_values": g_values,
        "g_iso": round(np.mean(g_values), 5) if g_values else None,
        "dHpp_G": round(dHpp, 2),
        "peak_height": round(float(signal[peak_idx]), 4),
        "trough_depth": round(float(signal[trough_idx]), 4),
        "second_integral": float(second_integral),
    }


# ---------------------------------------------------------------------------
# High-level workflow
# ---------------------------------------------------------------------------

def predict_and_compare_esr(exp_data: dict = None,
                            radical_type: str = "alkyl_radical",
                            mw_freq_ghz: float = 9.8,
                            journal: str = "jacs") -> dict:
    """
    Predict ESR parameters and, if experimental data provided, compare.
    """
    g_pred = predict_g_value(radical_type)
    hfc_pred = predict_hyperfine_pattern()

    result = {
        "g_prediction": g_pred,
        "hyperfine_prediction": hfc_pred,
    }

    if exp_data:
        g_exp = calculate_g_values(exp_data, mw_freq_ghz)
        result["g_experimental"] = g_exp

        # Plot experimental ESR
        field = np.array(exp_data.get("field_G", []))
        signal = np.array(exp_data.get("signal", []))

        if len(field) > 0:
            with ChemFigure(f"esr_{radical_type}", journal=journal) as cf:
                ax = cf.ax
                ax.plot(field, signal, color="#1a1a1a", linewidth=0.8)
                ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
                ax.set_xlabel("Magnetic Field / G")
                ax.set_ylabel("dI/dB")
                ax.set_title(
                    f"ESR — {radical_type}\n"
                    f"g = {g_exp.get('g_iso', '?')}  "
                    f"ν = {mw_freq_ghz} GHz  "
                    f"ΔH_pp = {g_exp.get('dHpp_G', '?')} G",
                    fontweight="normal", fontsize=8
                )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

    return result


# ---------------------------------------------------------------------------
# Spin trapping analysis
# ---------------------------------------------------------------------------

SPIN_TRAP_ADDUCTS = {
    "DMPO": {
        "full_name": "5,5-dimethyl-1-pyrroline N-oxide",
        "common_adducts": {
            "OH": {"aN": 14.9, "aH": 14.9, "description": "DMPO-OH (hydroxyl adduct)"},
            "OOH": {"aN": 14.3, "aH_beta": 11.7, "aH_gamma": 1.25, "description": "DMPO-OOH (superoxide adduct)"},
            "CH3": {"aN": 16.3, "aH": 23.5, "description": "DMPO-CH3 (methyl adduct)"},
            "SO3": {"aN": 14.7, "aH": 16.0, "description": "DMPO-SO3 (sulfite adduct)"},
        }
    },
    "PBN": {
        "full_name": "N-tert-butyl-α-phenylnitrone",
        "common_adducts": {
            "OH": {"aN": 14.8, "aH": 2.8, "description": "PBN-OH"},
            "CH3": {"aN": 15.0, "aH": 3.5, "description": "PBN-CH3"},
        }
    },
    "DEPMPO": {
        "full_name": "5-(diethoxyphosphoryl)-5-methyl-1-pyrroline N-oxide",
        "common_adducts": {
            "OH": {"aN": 14.0, "aH": 13.2, "aP": 47.2, "description": "DEPMPO-OH"},
            "OOH": {"aN": 13.3, "aH_beta": 11.0, "aP": 51.2, "description": "DEPMPO-OOH"},
        }
    },
}


def analyze_spin_trap(exp_data: dict, spin_trap: str = "DMPO",
                      mw_freq_ghz: float = 9.8) -> dict:
    """
    Analyze spin-trapping ESR data.
    Compares observed coupling constants with known adduct database.
    """
    trap_info = SPIN_TRAP_ADDUCTS.get(spin_trap.upper(), {})
    g_exp = calculate_g_values(exp_data, mw_freq_ghz)

    # Simple matching: compare g-value and estimate couplings from peak positions
    field = np.array(exp_data.get("field_G", []))
    signal = np.array(exp_data.get("signal", []))

    # Estimate hyperfine couplings from peak spacings
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal, height=np.max(signal) * 0.1)
    neg_peaks, _ = find_peaks(-signal, height=np.max(-signal) * 0.1)

    all_extrema = sorted(list(peaks) + list(neg_peaks))

    # Compute average spacing as coupling estimate
    spacings = []
    for i in range(len(all_extrema) - 1):
        spacings.append(abs(field[all_extrema[i+1]] - field[all_extrema[i]]))

    median_spacing = float(np.median(spacings)) if spacings else 0

    # Match to known adducts
    best_match = None
    best_score = float("inf")
    for adduct_name, params in trap_info.get("common_adducts", {}).items():
        aN = params.get("aN", 0)
        score = abs(median_spacing - aN)
        if score < best_score:
            best_score = score
            best_match = adduct_name

    return {
        "spin_trap": spin_trap,
        "trap_full_name": trap_info.get("full_name", ""),
        "g_experimental": g_exp,
        "estimated_coupling_G": round(median_spacing, 2),
        "best_adduct_match": best_match,
        "adduct_details": trap_info.get("common_adducts", {}).get(best_match, {}) if best_match else {},
        "all_known_adducts": list(trap_info.get("common_adducts", {}).keys()),
    }


print("[esr_processor] Ready.")
