# OpenVSP Controller

Questo programma ti permette di:

- Caricare il tuo modello di aereo (file `.vsp3`).
- Calcolare automaticamente le prestazioni di volo nella configurazione scelta.
- Valutare configurazioni diverse e condurre un processo di ottimizzazione sulle singole variabili di interesse.

---

## Requisiti

| Requisito | Versione |
| --- | --- |
| **Sistema Operativo** | Windows 10 / 11 |
| **Python** | **Esattamente la 3.13** (si crea automaticamente all'avvio) |
| **OpenVSP** | La cartella `OpenVSP-3.48.2-win64/` è già inclusa nel progetto |

> **Attenzione:** Versioni di Python 3.12 o precedenti **non** funzioneranno con l'eseguibile incluso.

---

## Guida Rapida

### Come avviarlo

1. Apri la cartella del progetto.
2. Fai **doppio click** sul file `run_project.bat`.
3. Aspetta qualche secondo.

Si aprirà automaticamente il quaderno di lavoro (notebook) con tutti i risultati (grafici, numeri, tabelle). Non devi installare niente: tutto è già pronto all'interno della cartella.

Questo singolo comando eseguirà in automatico i seguenti passaggi:

- Crea un ambiente virtuale locale in `.venv/`.
- Copia un ambiente Python 3.13 compatibile nella cartella `.venv/interpreter/`.
- Installa tutte le dipendenze elencate nel file `requirements.txt` all'interno della cartella `.venv/libs/`.
- Registra un kernel per Jupyter chiamato **OpenVSP Controller (.venv)**.
- Avvia lo script `scripts/verify_setup.py` per assicurarsi che tutto sia configurato correttamente.

### Apertura del Notebook Principale

JupyterLab si aprirà direttamente sul file `notebooks/main_analysis.ipynb`.  
La prima cella di codice ha il compito di avviare l'ambiente e verificare il corretto funzionamento della versione di OpenVSP inclusa.

---

## Altri Comandi

| Comando | Descrizione |
| --- | --- |
| `run_project.bat setup` | Crea da zero o ripara l'ambiente di lavoro locale |
| `run_project.bat` | Avvia JupyterLab aprendo direttamente il notebook principale |
| `run_project.bat verify` | Esegue solamente lo script di verifica `scripts/verify_setup.py` |

---

## Flusso di Lavoro Tipico

1. **Controllo dell'ambiente:** La primissima cella verifica la versione di Python, la posizione di OpenVSP e la corretta installazione dei pacchetti.
2. **Caricamento del modello:** Modifica la variabile `MODEL_PATH` inserendo il nome del tuo file `.vsp3` salvato dentro `models/`.
3. **Configurazione di base:** Imposta i valori per Mach, Reynolds, angolo d'attacco (alpha) e i parametri di stabilità all'interno della sezione `BASELINE_ANALYSIS`.
4. **Analisi di base:** Esegue una singola simulazione in VSPAERO al variare dell'angolo d'attacco, con la possibilità di leggere i file `.history` per la convergenza e `.stab` per la stabilità.
5. **Analisi delle singole variabili:** Studia in che modo lo spostamento della coda e del baricentro (CG) influenzano il margine statico e il punto neutro.
6. **Ottimizzazione:** Scegli quale metodo usare tra gradiente (SLSQP), ricerca Bayesiana (Optuna TPE) o un'ottimizzazione in due fasi.
7. **Esportazione:** I risultati vengono salvati nella cartella `exports/`: le tabelle in formato `.csv` e `.txt`, mentre i grafici come immagini `.png`.

---

## Dove Inserire il Tuo Modello

Metti il tuo file `.vsp3` all'interno della cartella `models/` e aggiorna la variabile `MODEL_PATH` nel notebook in questo modo:

```python
MODEL_PATH = REPO_ROOT / "models" / "il_tuo_aereo.vsp3"

## Struttura progetto

OpenVspController/
├── run_project.bat          ← Punto di accesso principale del programma
├── requirements.txt
├── environment.yml          ← File opzionale per l'uso con Conda
├── vspopt/                  ← Codice sorgente Python per analisi e ottimizzazione
├── notebooks/               ← Cartella dei notebook Jupyter
├── scripts/                 ← Script utili per l'installazione e la verifica
├── models/                  ← INSERISCI QUI I TUOI FILE .vsp3
├── exports/                 ← Qui troverai le tabelle e i grafici generati
├── OpenVSP-3.48.2-win64/    ← Versione di OpenVSP inclusa (non eliminare!)
├── .venv/                   ← Ambiente virtuale (creato in automatico dal setup)
├── interpreter/             ← Python 3.13 locale (creato in automatico dal setup)
└── libs/                    ← Pacchetti installati (creati in automatico dal setup)

## License

MIT
