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
- `interpreter/`: a repository-local copy of a compatible Python runtime, kept in Git
- `libs/`: third-party packages installed with `pip --target`, kept in Git
- `.venv/`: a local virtual environment created on top of `interpreter/`, kept in Git
- `results/`, `exports/`: solver outputs and derived artifacts kept as reference material when available

This is intentionally heavier than a source-only repository.
The tradeoff is deliberate: if bootstrap fails on another machine, an expert
user can inspect what is already present in `interpreter/`, `libs/`, `.venv/`,
and the latest output folders before trying to rebuild everything.

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

## Contributing

The repository is intentionally pinned to the bundled OpenVSP build.

If you update the OpenVSP folder in a future contribution, verify at minimum:

- the Python ABI required by the new `_vsp.pyd`;
- `scripts/verify_setup.py`;
- `run_project.ps1`;
- the notebook bootstrap section;
- the baseline sweep and MassProp flow.
 
## License

MIT
