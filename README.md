# OpenVSP Controller
Questo programma ti permette di:
- Caricare il tuo modello di aereo (file `.vsp3`)
- Calcolare automaticamente come vola nella configurazione scelta
- Valutare configurazioni diverse e condurre un processo di ottimizzazione sulle singole variabili di interesse

---

## Requirements

| Requirement | Version |
|---|---|
| **OS** | Windows 10 / 11 |
| **Python** | **3.13 exactly** (si crea all'avvio) |
| **OpenVSP** | `OpenVSP-3.48.2-win64/` is already in the repo |

> Python 3.12 or earlier will **not** work with the included binary.

---

## Quick start


### Come avviarlo

1. Apri la cartella del progetto.
2. **Doppio click** sul file `run_project.bat`
3. Aspetta qualche secondo.

Si aprirà automaticamente il quaderno con tutti i risultati (grafici, numeri, tabelle).  
Non devi installare niente: tutto è già dentro la cartella.

This single command:
- copies a compatible Python 3.13 runtime into `interpreter/`
- creates a local virtual environment in `.venv/`
- installs all dependencies from `requirements.txt` into `libs/`
- registers a Jupyter kernel named **OpenVSP Controller (.venv)**
- runs `scripts/verify_setup.py` to confirm everything is wired correctly

### 3 — Open the main notebook


JupyterLab opens directly on `notebooks/main_analysis.ipynb`.  
The first cell bootstraps the environment and verifies the embedded OpenVSP runtime.

---

## Other commands

| Command | Description |
|---|---|
| `run_project.bat setup` | Bootstrap / repair the local environment |
| `run_project.bat` | Launch JupyterLab on the main notebook |
| `run_project.bat verify` | Run `scripts/verify_setup.py` only |


---

## Typical notebook workflow

1. **Environment check** — the bootstrap cell confirms Python version, OpenVSP root, and package availability.
2. **Load model** — point `MODEL_PATH` to your `.vsp3` file under `models/`.
3. **Configure baseline** — set Mach, Reynolds, alpha schedule, and stability flags in `BASELINE_ANALYSIS`.
4. **Run baseline sweep** — a single VSPAERO alpha sweep with optional `.history` convergence and `.stab` stability parsing.
5. **One-at-a-time sweeps** — explore the effect of tail position and CG shift on static margin and neutral point.
6. **Optimize** — choose gradient (SLSQP), Bayesian (Optuna TPE), or two-phase optimization.
7. **Export** — tables are written to `exports/` as `.csv` + `.txt`; figures as `.png`.

---

## Placing your aircraft model

Drop your `.vsp3` file in the `models/` directory and update `MODEL_PATH` in the notebook:

```python
MODEL_PATH = REPO_ROOT / "models" / "your_aircraft.vsp3"
```

---

## Project layout (top level)

```
OpenVspController/
├── run_project.bat          ← main entry point
├── requirements.txt
├── environment.yml          ← Conda fallback (optional)
├── vspopt/                  ← Python analysis & optimization package
├── notebooks/               ← Jupyter notebooks
├── scripts/                 ← setup & verification helpers
├── models/                  ← place your .vsp3 files here
├── exports/                 ← generated tables and figures
├── OpenVSP-3.48.2-win64/   ← bundled OpenVSP runtime (do not delete)
|── .venv/                   ← virtual environment (created by setup)
├── interpreter/             ← local Python 3.13 runtime (created by setup)
├── libs/                    ← pip-installed packages (created by setup)
```

## License

MIT
