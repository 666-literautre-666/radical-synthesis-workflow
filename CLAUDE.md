# Radical Synthesis Semi-Automated Research Workflow

## Project overview

This is a semi-automated research workflow for a graduate student (研一) in radical synthesis (自由基合成). The system assists with: suggesting reaction routes, predicting spectra (NMR/MS/ESR), processing experimental data, and generating publication-quality editable figures.

## Research workflow

When the user describes a research topic/molecule:

1. **Reaction route suggestion**: Use `scripts/reaction_predictor.py` to analyze the substrate and suggest radical reaction routes, initiators, catalysts, and conditions. Supplement with literature searches (WebSearch).

2. **Literature review**: Search for relevant papers, conditions, and precedents. The user uses Zotero for reference management.

3. **After synthesis succeeds**:
   - Predict NMR, MS, ESR spectra using the prediction modules
   - When user provides experimental data, process it and overlay comparison plots
   - All figures must be editable (SVG + PDF formats, raw data CSV)

## Key modules

| Module | Purpose |
|--------|---------|
| `scripts/reaction_predictor.py` | Substrate analysis, radical initiator/catalyst recommendation, retrosynthesis |
| `scripts/nmr_processor.py` | 1H/13C shift prediction, Bruker/Varian data reading (nmrglue), peak picking |
| `scripts/ms_processor.py` | Isotopic pattern, fragment prediction, MS spectrum plotting |
| `scripts/esr_processor.py` | g-value prediction, hyperfine simulation, spin-trapping analysis |
| `scripts/plot_utils.py` | ChemFigure context manager, publication-quality plotting, journal styles |

## Plotting conventions (CRITICAL)

- **Always use `ChemFigure` context manager** for figures intended for publication
- **Always save as SVG + PDF** — the user edits figures in Illustrator/Inkscape
- **Always save raw data as CSV** alongside figures for Origin/GraphPad
- **Save pickle** for re-editing within Python
- Default journal style: JACS. Also available: `angewandte`, `nature_chem`, `acs`, `rsc`
- Journal style sheets are in `styles/` with consistent typography and sizing
- Single column: 3.3 in, Double: 7.0 in, aspect ratio ~golden ratio
- All plotting experiences are logged to `data/plot_experience.jsonl` for self-evolution

## Self-evolution (plotting memory)

After each plotting session, review `data/plot_experience.jsonl` and `scripts/plot_utils.py:get_plot_history()` to learn from past plots. The goal is to generate increasingly accurate, publication-ready figures that match top journal requirements. When the user gives feedback on a figure (e.g., "font too small", "wrong axis range"), update the plotting approach and document the lesson.

## Python environment

- Python 3.14 at `python3` (python3.exe)
- Default `python` command is Python 2.7 — always use `python3`
- pip is associated with Python 3
- Key packages: rdkit, nmrglue, matplotlib, seaborn, pyteomics, pandas, scipy, scikit-learn, openpyxl

## EasySpin + Octave EPR/ZFS fitting

The user has GNU Octave 10.2.0 + EasySpin 6.x running on this computer for EPR simulation and ZFS fitting:

- **Octave CLI**: `C:\Program Files\GNU Octave\Octave-10.2.0\mingw64\bin\octave-cli.exe`
- **EasySpin**: `C:\Users\xushaobo\easyspin\EasySpin-main\easyspin\`
- **EasySpin private**: `C:\Users\xushaobo\easyspin\EasySpin-main\easyspin\private\`
- **Octave startup**: `C:\Users\xushaobo\.octaverc` (adds EasySpin to path)
- **Compatibility patches applied**: datetime.m, verLessThan.m, split.m, pad.m in private/; source edits to pepper.m, garlic.m, saffron.m, easyspin_compile.m
- **8 MEX files compiled** for Octave: cubicsolve, lisum1i, projectzones, projecttriangles, multinucstick, multimatmult_, chili_lm, sf_peaks
- **ZFS fitting bridge**: `scripts/zfs_fitter.py` — `fit_zfs_from_csv('data.csv', S=1, mw_freq=9.5)` runs esfit and returns D, E, g parameters
- **Key constraint**: Octave's private directory MUST be explicitly added to path for .mex files to work
- **Desktop package**: `C:\Users\xushaobo\Desktop\EasySpin-Octave-ZFS\` and `EasySpin-Octave-ZFS-完整包.md` contain the full handoff package for colleagues

When the user asks about ZFS fitting / EPR simulation / EasySpin, use these paths. For fitting, generate an Octave script → run with octave-cli --no-gui → parse results.

## The user

- First-year graduate student (研一), field: radical synthesis (自由基合成)
- Has Aspen Plus/HYSYS v12 for process simulation
- Uses Zotero for reference management
- Prefers figures editable in Illustrator
- Works with: organic radicals, spin trapping (DMPO/PBN/DEPMPO), photoredox catalysis, ATRP
- Communicates via both VSCode IDE (this session) AND WeChat via cc-connect (separate sessions — the WeChat agent needs this CLAUDE.md to understand the local environment)
