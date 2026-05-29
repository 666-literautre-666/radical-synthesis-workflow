"""
Reaction route prediction for radical synthesis.
Combines RDKit-based retrosynthetic analysis with domain-specific
radical chemistry knowledge (initiators, mediators, catalysts).
"""
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFMCS
from rdkit.Chem import BRICS, Recap

from scripts.plot_utils import PREDICTIONS_DIR


# ---------------------------------------------------------------------------
# Radical chemistry knowledge base
# ---------------------------------------------------------------------------

RADICAL_INITIATORS = {
    "AIBN": {
        "name": "Azobisisobutyronitrile (AIBN)",
        "structure": "CC(C)(C#N)N=NC(C)(C)C#N",
        "t_half_10h": 65,        # °C in toluene
        "working_range": "50-80 °C",
        "solubility": "organic solvents",
        "byproduct": "N2 + tetramethylsuccinonitrile",
        "notes": "Most common azo initiator. Generates carbon-centered radicals.",
    },
    "BPO": {
        "name": "Benzoyl peroxide (BPO)",
        "structure": "O=C(OOC(=O)c1ccccc1)c1ccccc1",
        "t_half_10h": 73,
        "working_range": "60-90 °C",
        "solubility": "organic solvents",
        "byproduct": "CO2 + benzoic acid",
        "notes": "Peroxide initiator. Generates benzoyloxy and phenyl radicals.",
    },
    "DTBP": {
        "name": "Di-tert-butyl peroxide (DTBP)",
        "structure": "CC(C)(C)OOC(C)(C)C",
        "t_half_10h": 125,
        "working_range": "110-150 °C",
        "solubility": "organic solvents",
        "byproduct": "acetone + ethane (via t-BuO. → CH3. + acetone)",
        "notes": "High-temperature initiator. Generates t-BuO. radicals.",
    },
    "K2S2O8": {
        "name": "Potassium persulfate (KPS)",
        "t_half_10h": "N/A (thermal: 60-80 °C)",
        "working_range": "60-80 °C (aqueous)",
        "solubility": "water",
        "byproduct": "SO4²⁻",
        "notes": "Water-soluble inorganic initiator. Generates SO4.⁻ radicals.",
    },
    "TBHP": {
        "name": "tert-Butyl hydroperoxide (TBHP)",
        "structure": "CC(C)(C)OO",
        "t_half_10h": 170,
        "working_range": "120-160 °C, or RT with reducing agent",
        "solubility": "organic / aqueous",
        "byproduct": "t-BuOH",
        "notes": "Can be used with Fe(II) (Fenton-type) for low-T initiation.",
    },
    "ACCN": {
        "name": "1,1'-Azobis(cyclohexanecarbonitrile) (ACCN)",
        "structure": "N#CC1(CCCCC1)N=NC1(CCCCC1)C#N",
        "t_half_10h": 88,
        "working_range": "80-100 °C",
        "solubility": "organic solvents",
        "byproduct": "N2",
        "notes": "Higher-T alternative to AIBN.",
    },
}

CATALYSTS_MEDIATORS = {
    "Cu(I)/PMDETA": {
        "type": "ATRP catalyst",
        "metal": "Cu",
        "ligand": "PMDETA (N,N,N',N'',N''-pentamethyldiethylenetriamine)",
        "conditions": "RT to 80 °C, deoxygenated solvent",
        "notes": "Standard ATRP system. [Cu]/[L] = 1:1. Requires halide initiator (e.g., EBiB).",
    },
    "Cu(I)/TPMA": {
        "type": "ATRP catalyst",
        "metal": "Cu",
        "ligand": "TPMA (tris(2-pyridylmethyl)amine)",
        "conditions": "RT to 60 °C",
        "notes": "Highly active ATRP catalyst. Suitable for aqueous ATRP.",
    },
    "Ru(bpy)3": {
        "type": "Photoredox catalyst",
        "metal": "Ru",
        "ligand": "2,2'-bipyridine",
        "conditions": "visible light (450 nm), RT",
        "notes": "Classic photoredox catalyst. E°* = -0.81 V (strong reductant in excited state).",
    },
    "Ir(ppy)3": {
        "type": "Photoredox catalyst",
        "metal": "Ir",
        "ligand": "2-phenylpyridine",
        "conditions": "visible light (380-400 nm), RT",
        "notes": "Strong photoredox catalyst. E°* = -1.73 V (very strong excited state reductant).",
    },
    "Fe(acac)3": {
        "type": "Radical mediator / catalyst",
        "metal": "Fe",
        "conditions": "RT to 80 °C, various solvents",
        "notes": "Iron-catalyzed radical reactions. Cheap, low toxicity.",
    },
    "TEMPO": {
        "type": "Nitroxide mediator (NMP)",
        "structure": "CC1(C)CCCC(C)(C)N1[O]",
        "conditions": "120-140 °C",
        "notes": "Stable radical. Used in NMP (nitroxide-mediated polymerization) and as radical trap.",
    },
}

RADICAL_REACTION_TYPES = {
    "atom_transfer": {
        "description": "Atom transfer radical addition (ATRA) / cyclization",
        "typical_conditions": "Cu(I) cat., 60-80 °C, deoxygenated solvent",
        "substrates": "alkyl halide + alkene",
        "key_literature": "Curran, D.P. Synthesis 1988; Pintauer, T. Chem. Soc. Rev. 2008",
    },
    "hydrogen_atom_transfer": {
        "description": "HAT (hydrogen atom transfer)",
        "typical_conditions": "Peroxide initiator, light or heat, H-atom donor",
        "substrates": "C-H bond + radical acceptor",
        "key_literature": "Capaldo, L. et al. Chem. Rev. 2022",
    },
    "single_electron_transfer": {
        "description": "SET (single electron transfer) radical reaction",
        "typical_conditions": "Photoredox cat., visible light, RT",
        "substrates": "electron-rich + electron-poor partners",
        "key_literature": "Prier, C.K. et al. Chem. Rev. 2013",
    },
    "radical_addition": {
        "description": "Radical addition to unsaturated bonds",
        "typical_conditions": "Initiator + alkene/alkyne, 60-100 °C",
        "substrates": "R. + alkene/alkyne",
        "key_literature": "Zard, S.Z. Radical Reactions in Organic Synthesis, 2003",
    },
    "radical_cyclization": {
        "description": "Radical cyclization (5-exo-trig, 6-endo-trig, etc.)",
        "typical_conditions": "Bu3SnH/AIBN (classic) or photoredox (modern)",
        "substrates": "unsaturated halide/selenide",
        "key_literature": "Giese, B. Radicals in Organic Synthesis, 1986",
    },
    "homolytic_substitution": {
        "description": "Homolytic aromatic substitution (HAS)",
        "typical_conditions": "Peroxide, heat, or photoredox",
        "substrates": "aryl diazonium salt or aryl halide + nucleophile",
        "key_literature": "Studer, A. et al. Angew. Chem. 2016",
    },
}


# ---------------------------------------------------------------------------
# Reaction route suggestion
# ---------------------------------------------------------------------------

def analyze_substrate(smiles: str) -> dict:
    """
    Analyze a substrate molecule for radical reaction sites.
    Returns reactive sites ranked by predicted reactivity.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"error": f"Invalid SMILES: {smiles}"}

    info = {
        "smiles": smiles,
        "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
    }

    # Identify reactive sites
    reactive_sites = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        sym = atom.GetSymbol()
        aromatic = atom.GetIsAromatic()
        ring_info = atom.IsInRing()

        # C-X bonds (potential radical precursors)
        if sym in ["Br", "I", "Cl"]:
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == "C":
                    reactive_sites.append({
                        "site": f"C{idx}-{sym}",
                        "type": "carbon_halogen_bond",
                        "atom_index": idx,
                        "reactivity": "high" if sym in ["Br", "I"] else "moderate",
                        "suggested_reaction": "atom_transfer or radical_cyclization",
                        "note": f"C-{sym} bond: good radical precursor",
                    })

        # C=C double bonds (radical addition sites)
        if sym == "C":
            for bond in atom.GetBonds():
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    other = bond.GetOtherAtom(atom)
                    reactive_sites.append({
                        "site": f"C{idx}=C{other.GetIdx()}",
                        "type": "alkene",
                        "atom_index": idx,
                        "reactivity": "high",
                        "suggested_reaction": "radical_addition or ATRA",
                        "note": "C=C bond: radical addition / cyclization site",
                    })

        # Benzylic / allylic positions
        if sym == "C" and not aromatic:
            for nbr in atom.GetNeighbors():
                if nbr.GetIsAromatic():
                    h_count = atom.GetTotalNumHs()
                    if h_count > 0:
                        reactive_sites.append({
                            "site": f"C{idx} (benzylic)",
                            "type": "benzylic_C-H",
                            "atom_index": idx,
                            "reactivity": "high",
                            "suggested_reaction": "HAT or radical functionalization",
                            "note": f"Benzylic C-H: weak BDE ~85-90 kcal/mol",
                        })

        # Aldehyde C-H (very weak, BDE ~87 kcal/mol)
        if sym == "C":
            has_carbonyl_O = any(n.GetAtomicNum() == 8 and
                                 mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                                 for n in atom.GetNeighbors())
            has_H = atom.GetTotalNumHs() > 0
            if has_carbonyl_O and has_H:
                reactive_sites.append({
                    "site": f"C{idx} (aldehyde)",
                    "type": "aldehyde_C-H",
                    "atom_index": idx,
                    "reactivity": "high",
                    "suggested_reaction": "radical acylation or decarbonylation",
                    "note": "Aldehyde C-H: BDE ~87 kcal/mol, excellent HAT substrate",
                })

    info["reactive_sites"] = reactive_sites
    info["n_reactive_sites"] = len(reactive_sites)

    return info


def suggest_reaction_routes(smiles: str, target_transformation: str = "") -> dict:
    """
    Main workflow: given a substrate, suggest possible radical reaction routes.
    """
    substrate = analyze_substrate(smiles)

    suggestions = {
        "substrate_analysis": substrate,
        "target": target_transformation or "not specified",
    }

    # Match initiators based on substrate properties
    has_halogen = any(s["type"] == "carbon_halogen_bond" for s in substrate.get("reactive_sites", []))
    has_alkene = any(s["type"] == "alkene" for s in substrate.get("reactive_sites", []))
    has_benzylic = any(s["type"] == "benzylic_C-H" for s in substrate.get("reactive_sites", []))
    has_carbonyl = any(s["type"] == "aldehyde_C-H" for s in substrate.get("reactive_sites", []))

    recommended_initiators = []
    if has_halogen:
        recommended_initiators.extend(["AIBN", "ACCN"])
        suggestions["recommended_reaction_types"] = ["atom_transfer", "radical_cyclization"]
    if has_alkene:
        recommended_initiators.append("BPO")
        if "radical_addition" not in suggestions.get("recommended_reaction_types", []):
            suggestions.setdefault("recommended_reaction_types", []).append("radical_addition")
    if has_benzylic or has_carbonyl:
        recommended_initiators.extend(["DTBP", "TBHP"])
        suggestions.setdefault("recommended_reaction_types", []).append("hydrogen_atom_transfer")

    suggestions["recommended_initiators"] = list(set(recommended_initiators)) or ["AIBN"]

    # Suggest catalyst/mediator
    if has_halogen:
        suggestions["recommended_catalysts"] = ["Cu(I)/PMDETA", "Cu(I)/TPMA"]
        suggestions["catalyst_type"] = "ATRP-type"
    elif has_benzylic or has_carbonyl:
        suggestions["recommended_catalysts"] = ["Ru(bpy)3", "Ir(ppy)3"]
        suggestions["catalyst_type"] = "Photoredox"
    else:
        suggestions["recommended_catalysts"] = ["TEMPO", "Fe(acac)3"]

    # Suggest conditions
    suggestions["suggested_conditions"] = {
        "temperature": "60-80 °C (thermal initiator) or RT (photoredox)",
        "solvent": "MeCN, DCE, or PhH (deoxygenated, freeze-pump-thaw or N2 sparge)",
        "atmosphere": "N2 or Ar (strictly oxygen-free)",
        "concentration": "0.1-0.5 M substrate, initiator 10-20 mol%",
        "time": "Monitor by TLC or GC-MS, typically 2-24 h",
    }

    # Save
    pred_path = PREDICTIONS_DIR / f"{smiles[:30].replace(' ', '_')}_routes.json"
    with open(pred_path, "w") as f:
        json.dump(suggestions, f, indent=2, default=str)

    return suggestions


# ---------------------------------------------------------------------------
# Retrosynthetic analysis (BRICS-based)
# ---------------------------------------------------------------------------

def retrosynthetic_bonds(smiles: str) -> list[dict]:
    """
    Identify bonds that could be disconnected in a retrosynthetic sense.
    Uses BRICS fragmentation rules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []

    frags = BRICS.BreakBRICSBonds(mol)
    bonds = list(BRICS.FindBRICSBonds(mol))

    results = []
    for bond_indices, bond_type in bonds:
        a1, a2 = bond_indices[0], bond_indices[1]
        atom1 = mol.GetAtomWithIdx(int(a1))
        atom2 = mol.GetAtomWithIdx(int(a2))
        results.append({
            "bond": f"{atom1.GetSymbol()}{a1}-{atom2.GetSymbol()}{a2}",
            "bond_type": str(bond_type),
            "fragment_smiles": Chem.MolToSmiles(frags[0]) if frags else "",
        })

    return results


print("[reaction_predictor] Ready.")
