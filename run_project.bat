@echo off
setlocal

REM === MODIFICA: commento chiaro sulla nuova logica ===
REM Questo file è l'unico punto di ingresso del progetto.
REM Ora usa esclusivamente uv (velocissimo e affidabile).
REM Grazie al pin in pyproject.toml, uv scaricherà automaticamente Python 3.13
REM se non è già presente → risolve definitivamente il bug della versione sbagliata.

REM Imposta il percorso locale per uv (non sporca il sistema)
set "UV_BIN_DIR=%~dp0.uv_bin"
set "PATH=%UV_BIN_DIR%;%PATH%"

REM 1. Installa uv se non esiste
if not exist "%UV_BIN_DIR%\uv.exe" (
    echo [INFO] uv non trovato in locale. Installazione in corso...
    mkdir "%UV_BIN_DIR%"
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    copy "%USERPROFILE%\.local\bin\uv*.exe" "%UV_BIN_DIR%\" >nul
)

REM 2. Sincronizza ambiente (crea/aggiorna .venv con Python 3.13 esatto)
echo [INFO] Sincronizzazione ambiente con Python 3.13...
uv sync --frozen

REM 3. Avvia il wrapper (esegue il notebook e cattura errori)
echo [INFO] Esecuzione analisi notebook in corso...
uv run python run_wrapper.py

pause