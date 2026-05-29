r"""
EasySpin ZFS fitting bridge for EPR data analysis.
Runs Octave+EasySpin in batch mode to fit Zero Field Splitting parameters
(D, E) and hyperfine couplings from experimental CW-EPR spectra.

Workflow:
  1. Read experimental EPR data (CSV: field, intensity)
  2. Write exp data + Octave script to temp directory
  3. Run octave-cli --no-gui
  4. Parse fitted parameters + uncertainties
  5. Generate publication-quality overlay plot (exp vs fitted)
"""

import subprocess
import tempfile
import json
import os
import re
import time
from pathlib import Path
import numpy as np

from scripts.plot_utils import ChemFigure, FIGURES_DIR, EXPERIMENTAL_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OCTAVE_CLI = r"C:\Program Files\GNU Octave\Octave-10.2.0\mingw64\bin\octave-cli.exe"
EASYSPIN_DIR = r"C:\Users\xushaobo\easyspin\EasySpin-main\easyspin"
EASYSPIN_PRIVATE = os.path.join(EASYSPIN_DIR, "private")

EXPERIMENTAL_DIR.mkdir(parents=True, exist_ok=True)

# EasySpin path in Octave-friendly format
_EASYSPIN_POSIX = EASYSPIN_DIR.replace("\\", "/")
_EASYSPIN_PRIVATE_POSIX = EASYSPIN_PRIVATE.replace("\\", "/")


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

def check_environment() -> dict:
    """Check if Octave and EasySpin are available. Returns status dict."""
    result = {
        "octave_ok": False,
        "easyspin_ok": False,
        "octave_path": OCTAVE_CLI,
        "easyspin_path": EASYSPIN_DIR,
        "issues": [],
    }

    if not os.path.exists(OCTAVE_CLI):
        result["issues"].append(f"Octave not found at {OCTAVE_CLI}")
        return result
    result["octave_ok"] = True

    if not os.path.isdir(EASYSPIN_DIR):
        result["issues"].append(f"EasySpin not found at {EASYSPIN_DIR}")
        return result

    # Quick smoke test
    try:
        proc = subprocess.run(
            [OCTAVE_CLI, "--no-gui", "--eval",
             f"addpath('{_EASYSPIN_POSIX}'); addpath('{_EASYSPIN_PRIVATE_POSIX}'); "
             "disp(['ESOK:' num2str(exist('pepper')>0)]);"],
            capture_output=True, text=True, timeout=30, cwd=tempfile.gettempdir()
        )
        if "ESOK:1" in proc.stdout or "ESOK: 1" in proc.stdout:
            result["easyspin_ok"] = True
        else:
            result["issues"].append("EasySpin pepper() not callable")
    except Exception as e:
        result["issues"].append(f"Smoke test failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Read experimental data
# ---------------------------------------------------------------------------

def read_epr_csv(filepath: str, field_col: int = 0, intensity_col: int = 1,
                 skip_rows: int = 0, delimiter: str = None) -> dict:
    """
    Read experimental EPR data from CSV/TXT file.
    Returns dict with 'field_mT' and 'intensity' as numpy arrays.
    Auto-detects Gauss vs mT (converts if median field > 1000).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if delimiter is None:
        with open(filepath) as f:
            first_line = f.readline()
        delimiter = "," if "," in first_line else None

    data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skip_rows)
    field = data[:, field_col]
    intensity = data[:, intensity_col]

    if np.median(field) > 1000:
        field = field / 10.0
        print(f"[zfs_fitter] Detected field in Gauss -> converted to mT "
              f"(range: {field[0]:.1f}-{field[-1]:.1f})")

    return {
        "field_mT": field,
        "intensity": intensity,
        "source": str(path),
        "n_points": len(field),
    }


# ---------------------------------------------------------------------------
# Octave script generation (data via CSV, NOT inline)
# ---------------------------------------------------------------------------

OCTAVE_SCRIPT_TEMPLATE = r"""% Auto-generated ZFS fitting script by zfs_fitter.py
warning('off', 'all');

addpath('{easyspin_dir}');
addpath('{easyspin_private}');

% ---- Load experimental data from CSV ----
csv_data = csvread('{data_csv}');
field_mT = csv_data(:, 1);
exp_signal = csv_data(:, 2);

% ---- Experimental parameters ----
Exp.mwFreq = {mw_freq};
Exp.Range = [{field_min} {field_max}];
Exp.nPoints = {n_points};
Exp.Harmonic = 0;
Exp.Temperature = {temperature};

% ---- Simulation options ----
Opt.Verbosity = 0;
Opt.Method = '{opt_method}';
{nucs_line}

% ---- Starting parameters for fitting ----
Sys0.S = {S};
Sys0.D = [{D0} {E0}];
Sys0.g = [{g0}];
Sys0.lwpp = [{lwpp}];
{nucs0_line}
{A0_line}

% ---- Allowed variations ----
Vary.D = [{D_vary} {E_vary}];
Vary.g = [{g_vary}];
{nucs_vary_line}
{A_vary_line}

% ---- Fit options ----
FitOpt.Method = '{method}';
FitOpt.Verbosity = 1;
FitOpt.maxIterations = {max_iter};
FitOpt.TolFun = 1e-6;

% ---- Run fitting ----
diary('{log_file}');
diary on;
fprintf('=== ZFS FITTING START ===\n');
fprintf('Field range: %.2f - %.2f mT\n', Exp.Range);
fprintf('MW Freq: %.4f GHz\n', Exp.mwFreq);
fprintf('Spin S: %d\n', Sys0.S);
fprintf('Start D/E: [%.1f %.1f] MHz, g: %.3f, lwpp: [%.3f %.3f]\n', Sys0.D, Sys0.g, Sys0.lwpp);
fprintf('Method: %s, maxIter: %d\n', FitOpt.Method, FitOpt.maxIterations);

tic;
result = esfit(exp_signal, @{sim_func}, {{Sys0, Exp, Opt}}, {{Vary}}, FitOpt);
elapsed = toc;
fprintf('Elapsed: %.1f s\n', elapsed);
diary off;

% ---- Save results ----
save('{results_file}', 'result', 'field_mT', 'exp_signal', 'Exp', 'Sys0');

% ---- Write key results as JSON for easy Python parsing ----
fid = fopen('{json_file}', 'w');
fprintf(fid, '{{\n');
fprintf(fid, '  "D_fitted": [%.4f, %.4f],\n', result.argsfit{{1}}.D);
fprintf(fid, '  "g_fitted": %.6f,\n', result.argsfit{{1}}.g);
if isfield(result.argsfit{{1}}, 'lwpp')
  fprintf(fid, '  "lwpp_fitted": [%.4f, %.4f],\n', result.argsfit{{1}}.lwpp);
end
if isfield(result.argsfit{{1}}, 'A')
  fprintf(fid, '  "A_fitted": [');
  for ia = 1:numel(result.argsfit{{1}}.A)
    if ia>1, fprintf(fid, ', '); end
    fprintf(fid, '%.4f', result.argsfit{{1}}.A(ia));
  end
  fprintf(fid, '],\n');
end
fprintf(fid, '  "rmsd": %.6f,\n', result.rmsd);
fprintf(fid, '  "ssr": %.6f,\n', result.ssr);
fprintf(fid, '  "scale": %.6f,\n', result.scale);
fprintf(fid, '  "elapsed_s": %.1f,\n', elapsed);
pfit_str = sprintf('%.6f, ', result.pfit);
pfit_str = pfit_str(1:end-2);
fprintf(fid, '  "pfit": [%s],\n', pfit_str);
fprintf(fid, '  "pnames": {{');
for k = 1:numel(result.pnames)
  if k>1, fprintf(fid, ', '); end
  fprintf(fid, '"%s"', result.pnames{{k}});
end
fprintf(fid, '}},\n');
if isfield(result, 'pstd')
  pstd_str = sprintf('%.8f, ', result.pstd);
  pstd_str = pstd_str(1:end-2);
  fprintf(fid, '  "pstd": [%s],\n', pstd_str);
end
fprintf(fid, '  "n_iter": %d\n', numel(result.bestfithistory.rmsd));
fprintf(fid, '}}\n');
fclose(fid);

% ---- Generate fitted spectrum on experimental field grid ----
[B_fit, spc_fit] = {sim_func}(result.argsfit{{1}}, Exp, Opt);
csv_out = [field_mT(:) exp_signal(:) B_fit(:) spc_fit(:)];
csvwrite('{csv_file}', csv_out);

fprintf('=== ZFS FITTING DONE ===\n');
fprintf('D = [%.1f, %.1f] MHz\n', result.argsfit{{1}}.D);
fprintf('g = %.5f\n', result.argsfit{{1}}.g);
if isfield(result.argsfit{{1}}, 'A')
  fprintf('A = [%s] MHz\n', sprintf('%.1f ', result.argsfit{{1}}.A));
end
fprintf('|E/D| = %.4f\n', abs(result.argsfit{{1}}.D(2)/max(abs(result.argsfit{{1}}.D(1)), 1e-12)));
fprintf('RMSD = %.6f\n', result.rmsd);
"""


def generate_octave_script(exp_data: dict, output_dir: str, **kwargs) -> str:
    """
    Generate an Octave script for ZFS fitting.
    Writes experimental data to CSV (not inline) to avoid huge scripts.

    Parameters
    ----------
    exp_data : dict with 'field_mT' and 'intensity' arrays
    output_dir : str -- directory for output files

    Optional kwargs (with defaults):
        mw_freq=9.5          Microwave frequency, GHz
        S=1                  Spin quantum number
        D0=600, E0=60        Starting ZFS D, E (MHz)
        g0=2.0               Starting g-value
        lwpp=[0.0, 1.0]      [Gaussian, Lorentzian] linewidth (mT)
        D_vary=400           Allowed D variation
        E_vary=100           Allowed E variation
        g_vary=0.3           Allowed g variation
        temperature=298      Temperature (K)
        regime='solid'       'solid'→pepper, 'solution'→garlic, 'slow'→chili
        method='simplex fcn' esfit method
        max_iter=200         Max simplex iterations
        Nucs=None            Nuclei string e.g. '14N,1H'
        A0=None              Starting hyperfine [A1 A2 ...] MHz
        A_vary=None          Hyperfine variation per nucleus
    """
    field = np.asarray(exp_data["field_mT"])
    intensity = np.asarray(exp_data["intensity"])

    # Interpolate to uniform grid
    n_points = kwargs.get("n_points", 1024)
    field_min, field_max = field.min(), field.max()
    field_uniform = np.linspace(field_min, field_max, n_points)
    intensity_uniform = np.interp(field_uniform, field, intensity)

    # Write data CSV
    data_csv = os.path.join(output_dir, "exp_data.csv")
    np.savetxt(data_csv, np.column_stack([field_uniform, intensity_uniform]),
               delimiter=",", fmt="%.6f")

    # Simulation function
    regime = kwargs.get("regime", "solid")
    sim_func_map = {"solid": "pepper", "solution": "garlic", "slow": "chili"}
    sim_func = sim_func_map.get(regime, "pepper")

    # Hyperfine handling
    nucs = kwargs.get("Nucs", None)
    A0 = kwargs.get("A0", None)
    A_vary = kwargs.get("A_vary", None)

    nucs_line = ""
    nucs0_line = ""
    A0_line = ""
    nucs_vary_line = ""
    A_vary_line = ""

    if nucs and A0 is not None:
        nucs_line = f"Sys0.Nucs = '{nucs}';"
        nucs0_line = f"SetNucsSys0 = {{'{nucs}'}};"
        A0_str = " ".join(str(a) for a in (A0 if isinstance(A0, list) else [A0]))
        A0_line = f"Sys0.A = [{A0_str}];"

        if A_vary is not None:
            Av_str = " ".join(str(a) for a in (A_vary if isinstance(A_vary, list) else [A_vary]))
            A_vary_line = f"Vary.A = [{Av_str}];"
            nucs_vary_line = "VaryAFitSet.Nucs = {'" + nucs.replace(",", "','") + "'};"
        else:
            A_vary_line = "Vary.A = [0];"

    script = OCTAVE_SCRIPT_TEMPLATE.format(
        easyspin_dir=_EASYSPIN_POSIX,
        easyspin_private=_EASYSPIN_PRIVATE_POSIX,
        data_csv=data_csv.replace("\\", "/"),
        mw_freq=kwargs.get("mw_freq", 9.5),
        field_min=round(field_min, 2),
        field_max=round(field_max, 2),
        n_points=n_points,
        temperature=kwargs.get("temperature", 298),
        S=kwargs.get("S", 1),
        D0=kwargs.get("D0", 600),
        E0=kwargs.get("E0", 60),
        g0=kwargs.get("g0", 2.0),
        lwpp=", ".join(str(x) for x in kwargs.get("lwpp", [0.0, 1.0])),
        D_vary=kwargs.get("D_vary", 400),
        E_vary=kwargs.get("E_vary", 100),
        g_vary=kwargs.get("g_vary", 0.3),
        sim_func=sim_func,
        opt_method=kwargs.get("opt_method", "matrix"),
        method=kwargs.get("method", "simplex fcn"),
        max_iter=kwargs.get("max_iter", 200),
        nucs_line=nucs_line,
        nucs0_line=nucs0_line,
        A0_line=A0_line,
        nucs_vary_line=nucs_vary_line,
        A_vary_line=A_vary_line,
        log_file=os.path.join(output_dir, "octave_log.txt").replace("\\", "/"),
        results_file=os.path.join(output_dir, "zfs_results.mat").replace("\\", "/"),
        json_file=os.path.join(output_dir, "zfs_params.json").replace("\\", "/"),
        csv_file=os.path.join(output_dir, "zfs_comparison.csv").replace("\\", "/"),
    )

    script_path = os.path.join(output_dir, "zfs_fit_script.m")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    return script_path


# ---------------------------------------------------------------------------
# Run Octave
# ---------------------------------------------------------------------------

def run_octave_fitting(script_path: str, timeout: int = 600) -> dict:
    """
    Run the Octave fitting script and parse results.
    Returns dict with fitted parameters; raises RuntimeError on failure.
    """
    if not os.path.exists(OCTAVE_CLI):
        raise RuntimeError(f"Octave CLI not found at {OCTAVE_CLI}")

    work_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)

    print(f"[zfs_fitter] Running Octave fitting...")
    print(f"  Work dir: {work_dir}")

    start = time.time()
    result = subprocess.run(
        [OCTAVE_CLI, "--no-gui", "--eval",
         f"cd('{work_dir}'); run('{script_name}')"],
        capture_output=True, text=True, timeout=timeout,
        cwd=work_dir,
    )
    elapsed = time.time() - start
    print(f"  Octave wall time: {elapsed:.1f} s")

    stderr = result.stderr or ""
    if stderr:
        errors = [l for l in stderr.split("\n")
                  if l.startswith("error:") and "warning:" not in l.lower()
                  and "shadows" not in l.lower()]
        if errors:
            raise RuntimeError(
                f"Octave errors:\n" + "\n".join(errors[:20]) +
                f"\n\nLast stderr lines:\n" + "\n".join(stderr.split("\n")[-10:])
            )

    # Parse JSON results
    json_path = os.path.join(work_dir, "zfs_params.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)

    # Fallback: parse stdout
    stdout = result.stdout or ""
    parsed = _parse_esfit_stdout(stdout)
    if parsed:
        return parsed

    raise RuntimeError(
        f"Could not parse esfit results.\n"
        f"Stdout (last 2000 chars):\n{stdout[-2000:]}"
    )


def _parse_esfit_stdout(stdout: str) -> dict:
    """Parse esfit results from stdout text (fallback)."""
    result = {}

    m = re.search(r"D\s*=\s*\[([\d.]+),\s*([\d.]+)\]\s*MHz", stdout)
    if m:
        result["D_fitted"] = [float(m.group(1)), float(m.group(2))]

    m = re.search(r"g\s*=\s*([\d.]+)", stdout)
    if m:
        result["g_fitted"] = float(m.group(1))

    m = re.search(r"RMSD\s*=\s*([\d.eE+-]+)", stdout)
    if m:
        result["rmsd"] = float(m.group(1))

    m = re.search(r"\|E/D\|\s*=\s*([\d.]+)", stdout)
    if m:
        result["e_over_d"] = float(m.group(1))

    m = re.search(r"Elapsed:\s*([\d.]+)\s*s", stdout)
    if m:
        result["elapsed_s"] = float(m.group(1))

    return result if "D_fitted" in result else None


# ---------------------------------------------------------------------------
# Standalone simulation (no fitting)
# ---------------------------------------------------------------------------

def simulate_zfs(S: int = 1, D: float = 800, E: float = 80,
                 g: float = 2.0023, mw_freq: float = 9.5,
                 field_range: tuple = (200, 500), n_points: int = 2048,
                 lwpp: tuple = (0.0, 0.5), regime: str = "solid",
                 Nucs: str = None, A: list = None,
                 temperature: float = 298) -> dict:
    """
    Simulate a ZFS EPR spectrum using EasySpin without fitting.

    Parameters
    ----------
    S : int -- spin quantum number
    D, E : float -- ZFS parameters (MHz)
    g : float -- isotropic g-value
    mw_freq : float -- microwave frequency (GHz)
    field_range : (min, max) -- sweep range (mT)
    n_points : int -- number of field points
    lwpp : (Gauss, Lorentz) -- linewidth components (mT)
    regime : 'solid' (pepper), 'solution' (garlic), 'slow' (chili)
    Nucs : str -- nuclei e.g. '14N' or '14N,1H'
    A : list -- hyperfine couplings (MHz), same order as Nucs
    temperature : float -- temperature (K)

    Returns
    -------
    dict with 'field_mT', 'intensity', 'params'
    """
    sim_func_map = {"solid": "pepper", "solution": "garlic", "slow": "chili"}
    sim_func = sim_func_map.get(regime, "pepper")

    # Auto-adjust n_points if Lorentzian linewidth is too narrow for field step
    field_span = field_range[1] - field_range[0]
    field_step = field_span / n_points
    if lwpp[1] > 0:
        lw_fwhm = lwpp[1] * 3 ** 0.5  # FWHM of Lorentzian derivative
        if lw_fwhm < 2 * field_step:
            n_points = max(n_points, int(2 * field_span * 2 / lw_fwhm))
            print(f"[zfs_fitter] Auto-adjusted n_points to {n_points} "
                  f"(Lorentzian FWHM {lw_fwhm:.4f} mT < 2*step {2*field_step:.4f})")

    work_dir = tempfile.mkdtemp(prefix="zfs_sim_")

    # Build Easyspin system
    nucs_setup = ""
    if Nucs:
        nucs_setup = f"Sys.Nucs = '{Nucs}';"
        if A:
            a_str = " ".join(str(a) for a in (A if isinstance(A, list) else [A]))
            nucs_setup += f"\nSys.A = [{a_str}];"

    sim_m = os.path.join(work_dir, "simulate.m")
    script = f"""warning('off', 'all');
addpath('{_EASYSPIN_POSIX}');
addpath('{_EASYSPIN_PRIVATE_POSIX}');

Sys.S = {S};
Sys.D = [{D} {E}];
Sys.g = {g};
Sys.lwpp = [{lwpp[0]} {lwpp[1]}];
{nucs_setup}

Exp.mwFreq = {mw_freq};
Exp.Range = [{field_range[0]} {field_range[1]}];
Exp.nPoints = {n_points};
Exp.Temperature = {temperature};

Opt.Verbosity = 0;

[B, spc] = {sim_func}(Sys, Exp, Opt);
csv_out = [B(:) spc(:)];
csvwrite('{os.path.join(work_dir, "simulated.csv").replace(chr(92), "/")}', csv_out);
fprintf('SIMDONE\\n');
"""
    with open(sim_m, "w", encoding="utf-8") as f:
        f.write(script)

    result = subprocess.run(
        [OCTAVE_CLI, "--no-gui", "--eval",
         f"cd('{work_dir}'); run('{os.path.basename(sim_m)}')"],
        capture_output=True, text=True, timeout=120, cwd=work_dir
    )

    if "SIMDONE" not in (result.stdout or ""):
        raise RuntimeError(
            f"Simulation failed.\nstderr: {result.stderr}\nstdout: {result.stdout}"
        )

    csv_path = os.path.join(work_dir, "simulated.csv")
    if not os.path.exists(csv_path):
        raise RuntimeError("Simulated CSV not produced")

    data = np.loadtxt(csv_path, delimiter=",")
    return {
        "field_mT": data[:, 0],
        "intensity": data[:, 1],
        "params": {
            "S": S, "D": D, "E": E, "g": g,
            "mw_freq": mw_freq, "regime": regime,
            "lwpp": list(lwpp), "temperature": temperature,
        },
        "work_dir": work_dir,
    }


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def fit_zfs_from_csv(csv_path: str, S: int = 1,
                     mw_freq: float = 9.5,
                     D0: float = 600, E0: float = 60,
                     g0: float = 2.0,
                     D_vary: float = 400, E_vary: float = 100,
                     lwpp: tuple = (0.0, 1.0),
                     regime: str = "solid",
                     method: str = "simplex fcn",
                     Nucs: str = None, A0: list = None,
                     A_vary: list = None,
                     journal: str = "jacs",
                     skip_rows: int = 0,
                     field_col: int = 0, intensity_col: int = 1,
                     **kwargs) -> dict:
    """
    Fit ZFS parameters (D, E) and optionally hyperfine from experimental EPR data.

    Parameters
    ----------
    csv_path : str
        Path to experimental EPR data (CSV/TXT: field, intensity).
    S : int
        Spin quantum number (1=triplet, 3/2, 2, 5/2, ...).
    mw_freq : float
        Microwave frequency in GHz (X-band ~9.5, Q-band ~34).
    D0, E0 : float
        Starting guess for D and E (MHz).
    g0 : float
        Starting guess for g-value.
    D_vary, E_vary : float
        Allowed variation range for D, E (MHz).
    lwpp : (float, float)
        Linewidth [Gaussian, Lorentzian] in mT.
    regime : str
        'solid'→pepper, 'solution'→garlic, 'slow'→chili.
    method : str
        'simplex fcn' (fast), 'simplex int' (accurate), 'levmar int', etc.
    Nucs : str
        Nuclei string e.g. '14N' or '14N,1H' for hyperfine fitting.
    A0 : list
        Starting hyperfine couplings (MHz).
    A_vary : list
        Allowed hyperfine variation per nucleus.
    journal : str
        Journal style for output figure.
    skip_rows : int
        Rows to skip at CSV start.
    field_col, intensity_col : int
        Column indices for field and intensity.

    Returns
    -------
    dict with:
        D_fitted, g_fitted, rmsd, ssr, elapsed_s,
        figure_paths, work_dir
    """
    # 0. Quick environment check
    env = check_environment()
    if not env["octave_ok"] or not env["easyspin_ok"]:
        raise RuntimeError(
            f"Environment not ready:\n" + "\n".join(env["issues"])
        )

    # 1. Read experimental data
    print(f"\n[zfs_fitter] Reading: {csv_path}")
    exp_data = read_epr_csv(csv_path, field_col=field_col,
                            intensity_col=intensity_col, skip_rows=skip_rows)
    print(f"  Field: {exp_data['field_mT'][0]:.1f} - {exp_data['field_mT'][-1]:.1f} mT, "
          f"{exp_data['n_points']} points")

    # 2. Create temp working directory
    work_dir = tempfile.mkdtemp(prefix="zfs_fit_")
    print(f"  Work dir: {work_dir}")

    # 3. Generate Octave script (data goes to CSV)
    kw = dict(
        mw_freq=mw_freq, S=S, D0=D0, E0=E0, g0=g0,
        D_vary=D_vary, E_vary=E_vary,
        lwpp=list(lwpp), regime=regime, method=method,
        Nucs=Nucs, A0=A0, A_vary=A_vary,
    )
    kw.update(kwargs)
    script_path = generate_octave_script(exp_data, work_dir, **kw)

    # 4. Run Octave fitting
    results = run_octave_fitting(script_path, timeout=kwargs.get("timeout", 600))

    # 5. Parse results
    D_fitted = results.get("D_fitted", [0, 0])
    g_fitted = results.get("g_fitted", 0)
    rmsd = results.get("rmsd", float("inf"))
    A_fitted = results.get("A_fitted", None)

    print(f"\n[zfs_fitter] === RESULTS ===")
    print(f"  D  = [{D_fitted[0]:.1f}, {D_fitted[1]:.1f}] MHz")
    print(f"  |E/D| = {abs(D_fitted[1] / max(abs(D_fitted[0]), 1e-12)):.4f}")
    print(f"  g  = {g_fitted:.5f}")
    if A_fitted:
        print(f"  A  = {A_fitted} MHz")
    if "lwpp_fitted" in results:
        print(f"  lwpp = {results['lwpp_fitted']} mT")
    print(f"  RMSD = {rmsd:.6f}")
    if "elapsed_s" in results:
        print(f"  Time: {results['elapsed_s']:.1f} s")
    if "pstd" in results and "pnames" in results:
        print(f"  Uncertainties:")
        for name, std in zip(results["pnames"], results["pstd"]):
            print(f"    {name}: ±{std:.4f}")

    # 6. Generate publication-quality figure
    csv_out = os.path.join(work_dir, "zfs_comparison.csv")
    mol_name = Path(csv_path).stem
    figure_paths = _plot_zfs_fit(csv_out, results, exp_data, mol_name, journal)

    results["figure_paths"] = figure_paths
    results["work_dir"] = work_dir
    results["exp_data"] = exp_data

    return results


def _plot_zfs_fit(csv_file: str, results: dict, exp_data: dict,
                  mol_name: str, journal: str) -> list:
    """Generate publication-quality overlay plot (exp vs fitted + residual)."""
    if not os.path.exists(csv_file):
        print("[zfs_fitter] No comparison CSV -- skipping plot")
        return []

    data = np.loadtxt(csv_file, delimiter=",")
    field_exp = data[:, 0]
    signal_exp = data[:, 1]
    field_fit = data[:, 2]
    signal_fit = data[:, 3]

    D = results.get("D_fitted", [0, 0])
    g = results.get("g_fitted", 0)
    rmsd = results.get("rmsd", float("inf"))

    residual = signal_exp - signal_fit

    with ChemFigure(f"zfs_fit_{mol_name}", journal=journal, width="single") as cf:
        ax = cf.ax
        ax.plot(field_exp, signal_exp, "k-", linewidth=1.0, label="Exp.")
        ax.plot(field_fit, signal_fit, "r--", linewidth=1.0, label="Fit")
        ax.plot(field_exp, residual, "b-", linewidth=0.5, alpha=0.5,
                label="Resid.")

        title = (
            f"ZFS Fit -- {mol_name}\n"
            f"D = [{D[0]:.1f}, {D[1]:.1f}] MHz  "
            f"|E/D| = {abs(D[1]/max(abs(D[0]), 1e-12)):.3f}  "
            f"g = {g:.4f}  "
            f"RMSD = {rmsd:.4f}"
        )
        ax.set_xlabel("Magnetic Field / mT")
        ax.set_ylabel("EPR Signal / arb. units")
        ax.set_title(title, fontweight="normal", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        saved = list(cf._saved_paths)

    print(f"[zfs_fitter] Figure: {mol_name}")
    return saved


# ---------------------------------------------------------------------------
# Demo (simulate + fit in one Octave call)
# ---------------------------------------------------------------------------

def demo():
    """
    End-to-end demo: simulate a ZFS spectrum with noise, fit it back.
    Uses CSV for data exchange, simplex fcn for speed.
    """
    print("\n[zfs_fitter] === DEMO: Simulate + Fit ZFS ===\n")

    env = check_environment()
    if not env["octave_ok"] or not env["easyspin_ok"]:
        print(f"  SKIP: Environment issues: {env['issues']}")
        return None

    work_dir = tempfile.mkdtemp(prefix="zfs_demo_")
    posix = lambda p: p.replace("\\", "/")

    demo_m = os.path.join(work_dir, "demo_onestep.m")
    script = f"""warning('off', 'all');
addpath('{posix(EASYSPIN_DIR)}');
addpath('{posix(EASYSPIN_PRIVATE)}');

% ---- Simulate "experimental" spectrum ----
SysTrue.S = 1; SysTrue.D = [880 68]; SysTrue.g = 2.0023; SysTrue.lwpp = [0.0 1.0];
Exp2048.mwFreq = 9.5; Exp2048.Range = [250 450]; Exp2048.nPoints = 2048;
Opt.Verbosity = 0;
[B, spc] = pepper(SysTrue, Exp2048, Opt);
rng(42);
spc = spc + 0.005 * max(abs(spc)) * randn(size(spc));

% Interpolate to fitting grid
npts = 256;
field_uni = linspace(B(1), B(end), npts)';
spc_uni = interp1(B, spc, field_uni, 'linear');

% ---- Fit with simplex fcn (fast) ----
Exp.mwFreq = 9.5; Exp.Range = [field_uni(1) field_uni(end)];
Exp.nPoints = npts; Exp.Harmonic = 0;

Sys0.S = 1; Sys0.D = [800 50]; Sys0.g = 2.0; Sys0.lwpp = [0.0 1.0];
Vary.D = [500 100]; Vary.g = 0.3;

FitOpt.Method = 'simplex fcn';
FitOpt.Verbosity = 1;
FitOpt.maxIterations = 100;
FitOpt.TolFun = 1e-5;

fprintf('True:  D=[880 68] g=2.0023\\n');
fprintf('Start: D=[600 60] g=2.0\\n');
fprintf('Method: simplex fcn, npts=%d\\n', npts);

tic;
result = esfit(spc_uni, @pepper, {{Sys0, Exp, Opt}}, {{Vary}}, FitOpt);
elapsed = toc;

fprintf('\\n=== FINAL ===\\n');
fprintf('D=[%.1f %.1f] MHz\\n', result.argsfit{{1}}.D);
fprintf('g=%.5f\\n', result.argsfit{{1}}.g);
fprintf('RMSD=%.6f\\n', result.rmsd);
fprintf('|E/D|=%.4f\\n', abs(result.argsfit{{1}}.D(2)/result.argsfit{{1}}.D(1)));
fprintf('Iterations: %d\\n', numel(result.bestfithistory.rmsd));
fprintf('Elapsed: %.1f s\\n', elapsed);

save('{posix(os.path.join(work_dir, 'demo_result.mat'))}', 'result');
fprintf('DEMO_DONE\\n');
"""
    with open(demo_m, "w", encoding="utf-8") as f:
        f.write(script)

    print("  Running Octave (simulate + fit with simplex fcn)...")
    result = subprocess.run(
        [OCTAVE_CLI, "--no-gui", "--eval",
         f"cd('{work_dir}'); run('{os.path.basename(demo_m)}')"],
        capture_output=True, text=True, timeout=600, cwd=work_dir
    )

    stdout = result.stdout or ""
    for line in stdout.split("\n"):
        stripped = line.strip()
        if any(kw in stripped for kw in ["D=[", "g=", "RMSD=", "|E/D|=",
                                           "Iterations:", "Elapsed:", "True:",
                                           "Start:", "Method:", "DEMO_DONE"]):
            print(f"  {stripped}")

    if result.returncode != 0:
        stderr = result.stderr or ""
        errors = [l for l in stderr.split("\n")
                  if "error:" in l and "warning:" not in l.lower()
                  and "shadows" not in l.lower()]
        if errors:
            print(f"  Errors: {errors}")
            return None

    print("\n[zfs_fitter] === DEMO done ===")
    return {"work_dir": work_dir, "stdout": stdout}


# ---------------------------------------------------------------------------
# Module init
# ---------------------------------------------------------------------------

print("[zfs_fitter] Ready.")
print("  fit_zfs_from_csv('your_epr_data.csv', S=1, mw_freq=9.5) -> fitted D, E, g + figure")
print("  simulate_zfs(S=1, D=800, E=80, regime='solid') -> simulated spectrum")
print("  check_environment() -> {octave_ok, easyspin_ok, issues}")
print("  demo() -> end-to-end test with simulated data")
