@echo off
setlocal

:: 1. Verifica se 'uv' è installato
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] 'uv' non trovato. Installazione in corso...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

:: 2. Sincronizza l'ambiente (scarica Python 3.13 e librerie se mancano)
echo [INFO] Sincronizzazione ambiente in corso...
uv sync

:: 3. Esegui il comando principale
:: Sostituisci 'app/main.py' con il tuo file di ingresso reale
echo [INFO] Avvio del progetto...
uv run python app/main.py %*

pause