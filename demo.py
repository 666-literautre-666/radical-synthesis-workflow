"""
Demo: run all modules to verify the semi-automated research workflow.
Each section generates editable figures (SVG/PDF) + data (CSV/pickle).
"""

import sys
import os
import numpy as np
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))  # allow running from project root

print("=" * 60)
print("Radical Synthesis — Semi-Automated Research Workflow Demo")
print("=" * 60)

# -------------------------------------------------------------------
# 1. Reaction route prediction
# -------------------------------------------------------------------
print("\n[1/5] Reaction route prediction...")
from scripts.reaction_predictor import analyze_substrate, suggest_reaction_routes

# Example: benzyl bromide radical chemistry
test_smiles = "BrCc1ccccc1"  # benzyl bromide
analysis = analyze_substrate(test_smiles)
print(f"  Substrate: {test_smiles}")
print(f"  Formula: {analysis['formula']}")
print(f"  MW: {analysis['molecular_weight']:.2f}")
print(f"  Reactive sites: {analysis['n_reactive_sites']}")
for site in analysis.get("reactive_sites", []):
    print(f"    - {site['site']} ({site['type']}): {site['note']}")

routes = suggest_reaction_routes(test_smiles, "radical cyclization with alkene")
print(f"  Recommended initiators: {routes['recommended_initiators']}")
print(f"  Recommended catalysts: {routes['recommended_catalysts']}")
print(f"  Conditions: {routes['suggested_conditions']['temperature']}")

# -------------------------------------------------------------------
# 2. NMR prediction
# -------------------------------------------------------------------
print("\n[2/5] NMR prediction...")
from scripts.nmr_processor import predict_nmr_from_smiles, predict_and_compare_nmr

nmr_pred = predict_nmr_from_smiles("CCO")  # ethanol as simple test
print(f"  Predicted {len(nmr_pred['proton_shifts'])} proton environments")
print(f"  Predicted {len(nmr_pred['carbon_shifts'])} carbon signals")
predict_and_compare_nmr("CCO", journal="jacs")
print("  → NMR figure saved to data/figures/")

# -------------------------------------------------------------------
# 3. MS prediction
# -------------------------------------------------------------------
print("\n[3/5] Mass spectrometry prediction...")
from scripts.ms_processor import predict_ms, plot_ms_prediction

ms_pred = predict_ms("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
print(f"  Formula: {ms_pred['formula']}")
print(f"  [M]: {ms_pred['monoisotopic_mass']:.2f}")
print(f"  [M+1] intensity: {ms_pred['intensity'][1]:.1f}%")
print(f"  Fragments found: {len(ms_pred['fragments'])}")
plot_ms_prediction("CC(=O)Oc1ccccc1C(=O)O", journal="jacs")
print("  → MS figure saved to data/figures/")

# -------------------------------------------------------------------
# 4. ESR prediction
# -------------------------------------------------------------------
print("\n[4/5] ESR prediction...")
from scripts.esr_processor import (
    predict_g_value, predict_hyperfine_pattern,
    predict_and_compare_esr, SPIN_TRAP_ADDUCTS
)

# Predict g-value for common radical types
for rtype in ["alkyl_radical", "nitroxide", "phenoxyl_radical"]:
    g = predict_g_value(rtype)
    print(f"  {g['description']:40s} g = {g['g_iso_predicted']}")

# Simulate hyperfine coupling
hfc = predict_hyperfine_pattern(nuclei=[
    {"nucleus": "14N", "coupling_G": 14.5, "multiplicity": 3, "n": 1},
    {"nucleus": "1H", "coupling_G": 2.5, "multiplicity": 2, "n": 3},
])
print(f"  Hyperfine pattern: {hfc['total_lines']} lines")

# Mock experimental ESR data and plot
mock_field = np.linspace(3450, 3550, 2048)
# Simulated nitroxide triplet (1:1:1) derivative lineshape
aN = 15.0  # Gauss
center = 3500.0
mock_signal = np.zeros_like(mock_field)
for i, offset in enumerate([-aN, 0, aN]):
    mock_signal += (mock_field - (center + offset)) / 2.0 * np.exp(
        -((mock_field - (center + offset)) / 1.5)**2
    ) * (-1 if i % 2 == 0 else 1)

exp_data = {"field_G": mock_field.tolist(), "signal": mock_signal.tolist()}
esr_result = predict_and_compare_esr(exp_data, radical_type="nitroxide",
                                     mw_freq_ghz=9.8, journal="jacs")
print(f"  g_iso from mock data: {esr_result.get('g_experimental', {}).get('g_iso')}")
print("  → ESR figure saved to data/figures/")

# Spin trapping
print(f"\n  Spin trap database: {list(SPIN_TRAP_ADDUCTS.keys())}")
for trap, info in SPIN_TRAP_ADDUCTS.items():
    adducts = list(info["common_adducts"].keys())
    print(f"    {trap}: {adducts}")

# -------------------------------------------------------------------
# 5. Publication-quality plotting (ChemFigure)
# -------------------------------------------------------------------
print("\n[5/5] Publication-quality figures...")
from scripts.plot_utils import ChemFigure, plot_kinetics, get_plot_history

# Kinetics plot (editable)
t = np.linspace(0, 60, 7)
conc = np.column_stack([
    0.1 * np.exp(-0.05 * t),        # [A] decay
    0.1 * (1 - np.exp(-0.05 * t)),  # [B] formation
])
plot_kinetics(t, conc, labels=["[Substrate]", "[Product]"],
              title="Radical Cyclization Kinetics")
from scripts.plot_utils import save_figure, FIGURES_DIR
fig, ax = plot_kinetics(t, conc, labels=["[Substrate]", "[Product]"],
                        title="Radical Cyclization Kinetics")
save_figure(fig, "kinetics_demo")
print("  → Kinetics figure saved to data/figures/")

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("Demo complete! Check these directories:")
print(f"  Figures:     {FIGURES_DIR}")
print(f"  Predictions: {Path('data/predictions')}")
print("=" * 60)
print("\nAll figures saved as: SVG (editable) + PDF + PNG + pickle")
print("Open .svg files in Illustrator/Inkscape to tweak.")
print("Open .pickle files in Python to re-edit programmatically.")
print("Raw data saved as .csv alongside each figure for Origin/GraphPad.")
