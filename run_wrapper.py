import subprocess
import datetime
import sys
import os
import asyncio

# ====================== MODIFICHE APPLICATE ======================
# 1. Imposta subito il corretto event loop Windows → elimina il warning zmq
#    che compariva in tutti i log del bug_report.
# 2. Stampa la versione Python usata (diagnostica immediata).
# 3. Messaggi più chiari per l'utente finale (non informatico).
# ================================================================

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

print(f"🚀 Avvio OpenVSP Controller con Python {sys.version.split()[0]}")
print(f"   (Interpreter: {sys.executable})\n")

# Percorsi dei file
NOTEBOOK_PATH = os.path.join("notebooks", "main_analysis.ipynb")
ERROR_LOG = "bug_report.txt"

def main():
    print(f"[1/2] Esecuzione di {NOTEBOOK_PATH} in background...")
    print("      Attendere prego. NON chiudere questa finestra.\n")

    comando = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--execute",
        "--inplace", 
        NOTEBOOK_PATH
    ]

    try:
        subprocess.run(comando, capture_output=True, text=True, check=True)
        print(f"[2/2] ✅ SUCCESSO! L'analisi è terminata.")
        print(f"      I risultati sono stati salvati dentro {NOTEBOOK_PATH}.")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ [ERRORE FATALE] L'esecuzione si è interrotta!")
        print(f"   Generazione del file di log: '{ERROR_LOG}'")

        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n========================================\n")
            f.write(f"DATA E ORA: {datetime.datetime.now()}\n")
            f.write(f"FILE: {NOTEBOOK_PATH}\n")
            f.write(f"PYTHON USATO: {sys.version}\n")
            f.write(f"========================================\n")
            f.write("--- DETTAGLIO ERRORE ---\n")
            f.write(e.stderr)
            f.write("\n--- OUTPUT DELLA CONSOLE ---\n")
            f.write(e.stdout)
            
        print("\n📄 Controlla il file bug_report.txt per i dettagli dell'errore.")
        print("   (Il problema più comune era la versione Python sbagliata - ora risolto)")

if __name__ == "__main__":
    main()