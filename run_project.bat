@echo off
setlocal

REM Imposta il percorso della cartella locale per uv
set "UV_BIN_DIR=%~dp0.uv_bin"
set "PATH=%UV_BIN_DIR%;%PATH%"

REM 1. Controlla se uv esiste
if not exist "%UV_BIN_DIR%\uv.exe" (
    echo [INFO] uv non trovato in locale. Installazione in corso...
    mkdir "%UV_BIN_DIR%"
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    copy "%USERPROFILE%\.local\bin\uv*.exe" "%UV_BIN_DIR%\"
)

REM 2. Sincronizzazione librerie (Scarichera nbconvert la prima volta)
echo [INFO] Sincronizzazione ambiente e librerie...
uv sync

REM 3. Avvia il wrapper che esegue il notebook e cattura gli errori
uv run python run_wrapper.py

pause