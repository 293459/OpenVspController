# OpenVSP Controller

**Il modo più semplice per analizzare e migliorare i tuoi aerei**

Questo programma ti permette di:
- Caricare il tuo modello di aereo (file `.vsp3`)
- Calcolare automaticamente come vola (forze, resistenza, portanza…)
- Ottimizzare il design (trova la forma migliore)

**Tutto in un solo click.**

### Come avviarlo

1. Apri la cartella del progetto.
2. **Doppio click** sul file `run_project.bat`
3. Aspetta qualche secondo.

Si aprirà automaticamente il quaderno con tutti i risultati (grafici, numeri, tabelle).  
Non devi installare niente: tutto è già dentro la cartella.

### Cosa succede dietro le quinte 

- Il file `run_project.bat` prepara tutto automaticamente (intepreti,librerie,virtual environment etc...).
- Usa un software `uv` che crea un ambiente pulito con Python 3.13 (la versione esatta richiesta da OpenVSP).
- Esegue l’analisi completa sul tuo modello tramite il file Jupyter notebook.
- Se qualcosa va storto, crea automaticamente un file `bug_report.txt` con tutti i dettagli.

