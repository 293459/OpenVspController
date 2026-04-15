import subprocess
import datetime
import sys
import os

# Percorsi dei file
NOTEBOOK_PATH = os.path.join("notebooks", "main_analysis.ipynb")
ERROR_LOG = "bug_report.txt"

def main():
    print(f"\n[1/2] Esecuzione di {NOTEBOOK_PATH} in background...")
    print("      Attendere prego. NON chiudere questa finestra.")

    # Comando per eseguire il notebook. 
    # '--inplace' sovrascrive il file originale salvandoci dentro i risultati.
    comando = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--execute",
        "--inplace", 
        NOTEBOOK_PATH
    ]

    try:
        # Esegue il notebook intercettando output ed eventuali bug
        subprocess.run(comando, capture_output=True, text=True, check=True)
        print(f"\n[2/2] SUCCESSO! L'analisi è terminata.")
        print(f"      I risultati sono stati salvati dentro {NOTEBOOK_PATH}.")

    except subprocess.CalledProcessError as e:
        # SE QUALCOSA VA STORTO: Intercetta il bug e salva il file di testo
        print(f"\n[ERRORE FATALE] L'esecuzione si è interrotta a causa di un bug!")
        print(f"Generazione del file di log: '{ERROR_LOG}'")

        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n========================================\n")
            f.write(f"DATA E ORA: {datetime.datetime.now()}\n")
            f.write(f"FILE: {NOTEBOOK_PATH}\n")
            f.write(f"========================================\n")
            f.write("--- DETTAGLIO ERRORE ---\n")
            f.write(e.stderr) # Questo contiene la riga esatta del crash
            f.write("\n--- OUTPUT DELLA CONSOLE ---\n")
            f.write(e.stdout)
            
        print("\nControlla il file bug_report.txt per i dettagli dell'errore.")

if __name__ == "__main__":
    main()