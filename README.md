# OpenVSP Controller

Automated aerodynamic analysis and optimization for aircraft models defined in
OpenVSP `.vsp3` files, using the official OpenVSP Python API.

## What changed

This repository is now designed around a simpler workflow:

- OpenVSP is already included in the repository under `OpenVSP-3.48.2-win64/`.
- The main entry point is `run_project.bat`.
- The setup now prepares three repository-local runtime folders:
  `interpreter/`, `libs/`, and `.venv/`.
- The setup step automatically runs `scripts/verify_setup.py`.
- The main notebook includes a bootstrap section for less experienced users.
- Baseline VSPAERO runs now expose extra diagnostics inspired by the legacy
  notebook workflow: reference quantities, MassProp data, convergence status,
  and stability derivatives when `.history` / `.stab` files are available.

## Portable repository layout

The project now follows this practical structure:

- `vspopt/`, `app/`, `notebooks/`, `scripts/`: your project code
- `interpreter/`: a **copy of Python 3.13 runtime** (stdlib + compiled extensions), created on first run
- `libs/`: **third-party packages installed with pip**, also created on first run
- `.venv/`: a **local virtual environment created on top of interpreter/**, also created on first run
- `results/`, `exports/`: solver outputs and derived artifacts kept as reference material when available

### Differences between interpreter/, libs/, and .venv/

| Folder | Purpose | Contains | Maintained by |
|--------|---------|----------|---|
| `interpreter/` | Portable Python 3.13 base runtime | Python executable, standard library, compiled extensions (e.g., `_socket.pyd`) | `bootstrap_python_runtime.py` on first setup |
| `libs/` | Packaged third-party dependencies | All packages from `requirements.txt` (numpy, pandas, scipy, matplotlib, etc.) | `run_project.bat setup` using pip |
| `.venv/` | Virtual environment wrapper | Symlinks/shortcuts to `interpreter/` + `libs/`, plus activation scripts | `run_project.bat setup` using venv module |
| `OpenVSP-3.48.2-win64/` | Bundled solver | OpenVSP binaries, Python API (`_vsp.pyd`), examples, VSPAERO | Committed to Git |

**Why three layers?**
- `interpreter/` is truly portable; it can be copied to another machine and work without recompilation
- `libs/` explicitly separates third-party packages from the base runtime, making it clear what was added
- `.venv/` is a convenience layer for Jupyter and tools that expect a virtual environment structure

**What gets committed to Git?**
- All three folders (`interpreter/`, `libs/`, `.venv/`) are committed to Git so the repository is reproducible
- This makes the repo larger but means an expert user can inspect or repair bootstrap failures

**First run:** When you clone the repository, these folders already exist (committed to Git) and are _ready to use immediately_.
If you need to rebuild them, run `run_project.bat setup` again.

## Git policy

The `.gitignore` now ignores only local noise that does not help reproduce or
debug the project:

- build/distribution folders such as `build/`, `dist/`, `.eggs/`, and `*.egg-info/`
- notebook checkpoints
- alternate local environment folders and bootstrap scratch space such as `venv/`, `env/`, `.conda/`, and `.tmp/`
- IDE/editor metadata, operating-system junk, and log files

What stays versioned on purpose:

- the bundled OpenVSP binaries and compiled Python extension modules
- the repository-local runtime folders `interpreter/`, `libs/`, and `.venv/`
- OpenVSP/VSPAERO working outputs such as `.history`, `.stab`, `.lod`, `.polar`, `.vspaero`, `.adb`, `*.csv_adb`, `*_DegenGeom.*`, and `*_massprop.csv`
- exported figures and documents written under `exports/`

At the moment `exports/` may still contain only `.gitkeep` if no reference
figure has been committed yet. The folder is still left visible so users know
where to expect generated plots.

## Why the bundled OpenVSP copy is a good idea

Yes, this logic makes sense.

Using the OpenVSP version already committed in the repository is better for
reproducibility than telling every user to download a different build. It keeps
the Python wrapper, the executable, and the project code aligned to one known
solver version. If a future contributor wants to update OpenVSP, they can do it
 in a controlled way and adapt the code only once for everyone.

## Important Python note

The bundled OpenVSP binary is not a pure Python package. It includes compiled
extension modules, and those are tied to a specific Python ABI.

In this repository the embedded `_vsp.pyd` binary references `python313.dll`,
so Python 3.13 is required for the local runtime.

That means:

- newer Python is not automatically backward-compatible here;
- Python 3.12 will not work with this bundled OpenVSP build;
- Python 3.13 is the minimum and exact version line required by the included
  OpenVSP package.

## Recommended setup

### 1. Make sure Python 3.13 is installed

The repository setup script looks for a compatible interpreter in this order:

1. `python`
2. `py -3.13`

If neither resolves to Python 3.13, the script stops and tells you what is
missing.

The launcher also checks the most common Windows install paths for Python 3.13.
If your interpreter is installed elsewhere, you can point to it explicitly:

```bat
run_project.bat setup -PythonExe "C:\path\to\python.exe"
```

### 2. Run the project bootstrap

From the repository root, open a terminal **in Visual Studio Code** and run:

```bat
run_project.bat setup
```

**Important:** It is strongly recommended to run `run_project.bat` from the VSCode integrated terminal
(View → Terminal) rather than from the Windows command prompt or PowerShell on the desktop.
The VSCode terminal provides proper error messages and diagnostic output, making it much easier
to troubleshoot any issues that may arise during setup or execution.

This command:

- creates or refreshes `interpreter/` inside the repository by copying a
  compatible Python 3.13 runtime into the project folder;
- creates or refreshes `.venv/` on top of that local interpreter;
- installs or refreshes the Python dependencies from `requirements.txt` into `libs/`;
- wires both `interpreter/` and `.venv/` to see `libs/` automatically;
- registers a Jupyter kernel named `OpenVSP Controller (.venv)`;
- runs `scripts/verify_setup.py`.

This means the repository itself becomes the place where the usable runtime
lives, instead of scattering dependencies across the PC.

Because `interpreter/`, `libs/`, and `.venv/` are versioned, `run_project.bat
setup` is not only a first-time bootstrap command anymore: it is also the
repair/refresh path when a clone already contains most of the environment.

### 3. Open the notebook

```bat
run_project.bat
```

This launches JupyterLab directly on:

```text
notebooks/main_analysis.ipynb
```

The notebook itself starts with a bootstrap section and explains the next steps.
If setup fails, the first code cell now prints the launcher output directly so
the user can see the real cause immediately.

**Important note:** After the first successful run of `run_project.bat setup`, you do not need to
run it again unless:
- You modified `requirements.txt` and want to install new packages
- The virtual environment becomes corrupted or broken
- You want to rebuild everything from scratch

For normal notebook usage, simply run `run_project.bat` (without `setup`) to launch JupyterLab.
The environment is already configured and ready to use.

## Other commands

Verify only:

```bat
run_project.bat verify
```

Launch the Streamlit interface:

```bat
run_project.bat streamlit
```

## Terminal recommendation

**Always use VSCode's integrated terminal** (View → Terminal) instead of cmd.exe, PowerShell, or Windows bash when running `run_project.bat` commands.

**Why?**
- VSCode terminal captures error messages and diagnostic output properly, allowing you to see exactly what failed
- Windows bash and cmd.exe may suppress or mangle error output, making problems harder to diagnose
- If setup fails, running from VSCode will show the real cause immediately

If an error occurs while running from outside VSCode, try running the same command again from within VSCode's terminal to get clearer feedback.

## Notebook-first workflow

The notebook is now the recommended interface for the final user.

The first sections are intentionally guided:

1. bootstrap the local environment if needed;
2. verify the embedded OpenVSP runtime;
3. load the model;
4. inspect reference quantities and MassProp output;
5. run the baseline sweep and read convergence / stability diagnostics.

This keeps the project easier to use for someone who is not comfortable with
manual environment management.

## What is now automatic

The user no longer needs to:

- download OpenVSP separately;
- create a `.pth` file manually;
- install project dependencies into the global Python installation;
- remember the verification command separately;
- discover the notebook kernel by hand if `run_project.bat setup` has already
  been executed.

## Portability note

The project is now much closer to a repository-local portable setup than a
typical Python repository.

What is already portable:

- the OpenVSP solver folder is versioned in the repository;
- the project code is versioned in the repository;
- the copied Python runtime lives in `interpreter/`;
- third-party packages live in `libs/`;
- the convenience virtual environment lives in `.venv/`.
- the latest committed solver outputs and exported artifacts can also live in
  `results/` and `exports/`.

What is still machine-sensitive:

- the very first refresh from scratch still needs access to a compatible Python
  3.13 interpreter if `interpreter/` is missing or intentionally rebuilt;
- `.venv/` remains a convenience layer and is inherently more fragile than the
  plain `interpreter/` + `libs/` combination;
- Jupyter kernel registration is user-local, so `run_project.bat setup` may
  still need to be executed once per machine/account.

So the repository is now much more self-descriptive even before bootstrap
works, but it is still not a perfect zero-dependency installer for a completely
clean offline PC.

Also note that `.venv/` is mainly a convenience layer for Jupyter and tools.
The real portable base is `interpreter/` + `libs/`. Virtual environments are
still less relocatable by nature than a plain folder-based runtime.

If full offline portability is ever required, the next step would be to bundle:

- a pre-shipped Python 3.13 runtime inside `interpreter/`;
- a local wheelhouse for the required packages.

## Features

- Load a `.vsp3` model and inspect geometry parameters as Python objects
- Run VSPAERO alpha sweeps programmatically
- Run parameter sweeps on design variables
- Gradient-based optimization via `scipy`
- Bayesian optimization via `optuna`
- Two-phase optimization: Bayesian search followed by local refinement
- Notebook and Streamlit entry points
- Reference quantities and MassProp reporting
- Convergence parsing from `.history`
- Stability derivative parsing from `.stab`

## Main files

```text
OpenVspController/
|-- run_project.bat
|-- README.md
|-- requirements.txt
|-- environment.yml
|-- interpreter/
|-- libs/
|-- .venv/
|-- results/
|-- exports/
|-- scripts/
|   |-- run_project.ps1
|   |-- bootstrap_pip.py
|   |-- bootstrap_python_runtime.py
|   `-- verify_setup.py
|-- notebooks/
|   `-- main_analysis.ipynb
|-- app/
|   `-- streamlit_app.py
|-- vspopt/
|   |-- openvsp_runtime.py
|   |-- wrapper.py
|   |-- postprocess.py
|   `-- ...
`-- OpenVSP-3.48.2-win64/
```

## Conda fallback

If someone still prefers Conda, `environment.yml` is kept as a fallback.
However, the supported and documented path is the repository-local runtime
created by `run_project.bat`.

## About requirements.txt

The `requirements.txt` file defines all Python package dependencies and is still necessary.
It is used by `run_project.bat setup` to populate the `libs/` folder with third-party packages.

**If you modify `requirements.txt`** (e.g., to add a new package), you must run
`run_project.bat setup` again to refresh the packages in `libs/`. After that, you can
use the notebook or scripts immediately without further configuration.

## Troubleshooting

### OpenVSP version check shows 0.0.0

**Problem:** The notebook cell displaying `[FAIL] OpenVSP: OpenVSP 0.0.0 is older than required 3.48.0`.

**Explanation:** This occurred when the `vsp.GetVersionString()` function did not work correctly in the Python
environment. However, the repository bundles OpenVSP `3.48.2-win64`, which is the correct version.

**Solution:** The version check has been updated to:
1. Try `GetVersionString()` first if available
2. Extract the version from the bundled folder name `OpenVSP-3.48.2-win64` if GetVersionString() fails or returns invalid results
3. Display the diagnostic source (API, bundled folder name, or unknown) in the message

If you still see 0.0.0, run `scripts/verify_setup.py` to check the full environment, or review the diagnostic
message to see which method was used to detect the version.

### Duplicate component names in the model

**Problem:** When loading a `.vsp3` file, you may see warnings like:
```
WARNING: You have duplicate names. Some parts are being overwritten!
    -> DUPLICATE NAME FOUND: 'MeshGeom'
```

This happens when your OpenVSP model contains two or more geometry components with identical names.

**Explanation:** OpenVSP allows creating components with the same name, but Python dictionaries require unique keys.
The loading process automatically renames duplicates by appending `_2`, `_3`, etc. (e.g., `MeshGeom` becomes `MeshGeom_2`)
to ensure all components are accessible in Python.

**Diagnostics:** After loading, you can call:
```python
diagnostics = model.wrapper.get_component_diagnostics()
print(f"Total components in VSP: {diagnostics['total_components']}")
print(f"Unique names after dedup: {diagnostics['unique_names']}")
print(f"Has duplicates: {diagnostics['has_duplicates']}")
for warning in diagnostics['warnings']:
    print(f"  - {warning}")
```

This shows:
- How many geometry objects exist in the VSP model
- How many unique names were created in Python
- Which components were renamed due to duplication
- Details on how to fix duplicates in your model

**Solution:** To avoid confusion in the future:
1. Open your `.vsp3` file in OpenVSP
2. Identify and rename any components with identical names to unique names
3. Save the model
4. Reload in this notebook — the warnings should disappear

All components are still present and usable; the renaming only affects their Python names.

### VSPAERO sweep produces fewer alpha points than requested

**Problem:** The VSPAERO sweep returns only 1 alpha point when 26 were requested:
```
VSPAEROResults(M=0.20, Re=1.0e+06, alpha=[0.0,0.0] x 1 pts, L/D_max=0.00)
[WARNING] Only 1 alpha points in sweep. Results may be coarse; consider increasing AlphaNpts.
```

**Explanation:** This can occur due to several reasons:
1. **VSPAERO crash or timeout:** The solver may fail silently without generating complete results
2. **Incomplete mesh generation:** If the OpenVSP model fails geometry-to-mesh conversion
3. **Parameter mismatch:** If VSPAERO doesn't recognize the sweep parameter names correctly
4. **Model corruption:** If the `.vsp3` file or geometry is invalid for VSPAERO analysis

**Diagnostics:**
1. Check the wrapper logs — look for errors during the sweep setup
2. Check if `.history` and `.stab` files were generated in the `results/` folder
3. Run `scripts/verify_setup.py` to ensure the environment is correct

**Solutions to try:**
- Verify that the OpenVSP model geometry is valid: run `File → Validate Geometry` in OpenVSP
- Simplify the geometry: remove any invalid or problematic components and try again
- Check that VSPAERO binary exists: `OpenVSP-3.48.2-win64/bin/vspaero.exe` should be present
- Ensure the model has proper reference area (Sref) defined
- Try with a known-good model first (like a simple wing) to confirm the environment works

The improved diagnostics in this version will now alert you if the actual number of points differs from requested,
making it easier to identify incomplete runs.

## File Formats

### .vsp3 files (Aircraft geometry)

**Description:** Binary/XML-based format that stores the complete parametric 3D geometry of your aircraft.

**Contains:**
- Aircraft component definitions (wings, fuselage, tail, propellers, etc.)
- Detailed geometric parameters: wing span, chord, airfoil curves, dihedral, sweep, etc.
- Component positions and orientations
- Design variable definitions and parameter limits
- Mesh and analysis settings

**Created/Edited by:**
- OpenVSP GUI application
- This project via the OpenVSP Python API
- Manual editing (with caution) as XML

**Used for:**
- Defining the starting geometry for all analysis runs
- Design iteration and parameter optimization
- Creating baseline configurations before running VSPAERO

**Example:** `models/VESPA.vsp3` is the aircraft geometry for this project.

### .vspaero files and related outputs (Aerodynamic analysis results)

**Description:** Text-based or binary results from the VSPAERO solver (vortex lattice / panel method).

**Contains:**
- Computed aerodynamic coefficients: CL, CD, CM (lift, drag, moment)
- Induced and profile drag components (CDi, CDo)
- Pressure distributions
- Force/moment data in various reference frames
- One file per analysis run or sweep configuration

**Accompanying files:**
- `.history` — Convergence history (iteration count, residuals)
- `.stab` — Stability derivatives (roll, pitch, yaw derivatives)
- `.lod` — Load distribution data
- `.polar` — Aerodynamic polar data
- `.adb` — Database files with result summaries

**Generated by:**
- `model.wrapper.run_vspaero_sweep()` during analysis
- The VSPAERO executable bundled in `OpenVSP-3.48.2-win64/bin/`

**Used for:**
- Understanding aerodynamic performance (L/D, stall behavior, etc.)
- Extracting data for optimization algorithms
- Validating designs against performance requirements
- Plotting aerodynamic polars and performance curves

**Workflow:**
```
.vsp3 model (geometry)
    ↓ (VSPAERO analysis)
    → .vspaero files (aerodynamic results)
    → .history, .stab (supplementary data)
    ↓ (post-processing)
    → Plots, tables, optimization
```

## About requirements.txt and dependency management

**Current design:**

The `requirements.txt` file is **still necessary** and defines the third-party Python packages required by the
project (numpy, pandas, scipy, matplotlib, plotly, optuna, scikit-optimize, streamlit, etc.).

- When you run `run_project.bat setup`, it uses `pip install ... -r requirements.txt` to populate the `libs/` folder
- The packages in `libs/` are then wired into the virtual environment activated in `.venv/`
- If you modify `requirements.txt` (to add a new package, for example), you must run `run_project.bat setup` again

**Why keep it?**

- Explicit declaration of all dependencies and their minimum versions
- Reproducibility: anyone can see what packages the project needs
- Flexibility: users can modify `requirements.txt` if needed for their environment
- Standard practice in Python projects

**You do NOT need to edit `requirements.txt` unless:**
- You want to add a new package (e.g., a new visualization library)
- You need to constrain specific package versions
- You're on an environment with unusual package compatibility issues

## Setup lifecycle and when to run run_project.bat

**First time (fresh clone):**
```cmd
run_project.bat setup
```
This:
- Detects or creates a local Python 3.13 runtime in `interpreter/`
- Installs packages from `requirements.txt` into `libs/`
- Creates a virtual environment in `.venv/`
- Registers a Jupyter kernel named `OpenVSP Controller (.venv)`
- Runs verification checks

**After first setup (normal usage):**
```cmd
run_project.bat
```
This:
- Launches JupyterLab directly on the notebook (no setup needed)
- Uses the existing environment from the first run

**You only need to run `run_project.bat setup` again if:**
- You modify `requirements.txt` and want to install new packages
- The virtual environment becomes corrupted or broken
- You want to rebuild everything from scratch
- You're on a new machine/account (Jupyter kernel registration is local)

**Important note:**
Do not run `run_project.bat setup` every time.  After the first successful run, the `interpreter/`, `libs/`, and
`.venv/` folders are configured and ready to use. Continuous re-running is unnecessary and slows down your workflow.
The environment is truly local and portable — it lives in your repository, not in system directories.

## Folder structure explanation (interpreter, libs, .venv)

This project uses a three-layer local runtime approach for maximum portability and clarity:

| Layer | Purpose | Contains | Maintained by |
|-------|---------|----------|---|
| **`interpreter/`** | Base Python 3.13 runtime | Python executable, standard library, compiled extensions (e.g., `_socket.pyd`) | `run_project.bat setup` (copied from system Python 3.13) |
| **`libs/`** | Third-party packages | All packages from `requirements.txt` (numpy, pandas, scipy, matplotlib, plotly, optuna, etc.) | `run_project.bat setup` using `pip install -r requirements.txt` |
| **`.venv/`** | Virtual environment wrapper | Symlinks/shortcuts to `interpreter/` + `libs/`, plus activation scripts (activate.bat, activate.ps1, etc.) | `run_project.bat setup` using Python `venv` module |

### Why three layers?

1. **`interpreter/`:** Purely portable — can be zipped and moved to another Windows machine without recompilation
2. **`libs/`:** Explicitly separates third-party additions from the base runtime, making it clear what was installed
3. **`.venv/`:** A convenience layer for tools (Jupyter, IDEs) that expect a standard virtual environment structure

### What gets committed to Git?

All three folders are **committed to Git** to ensure the repository is fully reproducible:
- Clone the repo → all three folders exist → run `run_project.bat` immediately (no setup needed)
- This makes the repo larger but guarantees that an expert user can inspect or repair bootstrap failures
- The bundled OpenVSP binary is also committed for the same reason

### Example workflows

**Scenario 1: You clone the repo for the first time (Windows machine with Python 3.13 installed)**
```
1. git clone <repo>
2. run_project.bat setup          (creates interpreter/ if missing, wires libs/ and .venv/)
3. Jupyter kernel is registered
4. run_project.bat                (launches notebook immediately)
```

**Scenario 2: You have already cloned and set up, now you work normally**
```
1. Open VSCode
2. run_project.bat                (launches notebook, environment already ready)
3. Kernel is already registered, select it in the notebook
4. Run analysis
```

**Scenario 3: You want to add a new package (e.g., networkx for graph visualizations)**
```
1. Edit requirements.txt:  add networkx>=3.0
2. run_project.bat setup          (pip installs networkx into libs/)
3. run_project.bat                (use notebook with the new package)
```

**Scenario 4: Something is broken (environment corrupted)**
```
1. run_project.bat setup          (refresh everything)
2. If that doesn't work, delete interpreter/ and .venv/ manually
3. run_project.bat setup          (rebuild them from scratch)
```

## Contributing

### Understanding .vsp3 and VSPAERO output files

This project works with two main types of files:

**`.vsp3` files** (Aircraft geometry)
- Binary/XML format containing the complete parametric 3D geometry of your aircraft
- Stores wing spans, fuselage length, airfoil shapes, component positions, etc.
- Created and edited in OpenVSP GUI or via the OpenVSP Python API
- Example: `models/VESPA.vsp3`
- Used as the starting point for all analysis runs

**`.vspaero` files and related outputs** (Aerodynamic analysis results)
- Text-based results from the VSPAERO solver (vortex lattice method)
- Contains computed aerodynamic coefficients: CL, CD, CM, pressure distributions
- Usually accompanied by `.history` (convergence info) and `.stab` (stability derivatives)
- Generated by `model.wrapper.run_vspaero_sweep()` in the notebook
- Stored in temporary locations or committed to Git under `results/` for reference

**Workflow:**
1. Load a `.vsp3` model → defines your aircraft shape
2. Run VSPAERO → generates `.vspaero`, `.history`, `.stab` files
3. Parse results → extract aerodynamic coefficients for analysis and optimization

### Contributing to this project

If you update the OpenVSP folder in a future contribution, verify at minimum:

- the Python ABI required by the new `_vsp.pyd`;
- `scripts/verify_setup.py`;
- `run_project.ps1`;
- the notebook bootstrap section;
- the baseline sweep and MassProp flow.
 
## License

MIT
