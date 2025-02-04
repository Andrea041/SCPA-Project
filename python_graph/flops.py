import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Soglie per suddividere i grafici in base al numero di nonzeri
soglie_nonzeri = [10000, 100000, 500000, 1e6, 2.5e6, 1e7]  # Definisci i valori di soglia per raggruppare i grafici

# Directory contenente i file JSON (aggiorna con il percorso corretto)
dir_json = "."

# Leggi tutti i file JSON nella directory
dati = []
for file_name in os.listdir(dir_json):
    if file_name.endswith(".json"):
        try:
            num_thread = int(file_name.split("_")[-1].split(".")[0])  # Estrai il numero di thread dal nome del file
        except ValueError:
            print(f"Skipping file {file_name} due to invalid thread number.")
            continue  # Salta i file che non contengono un numero valido

        with open(os.path.join(dir_json, file_name), "r") as f:
            file_dati = json.load(f)
            
            # Controlla se il file JSON è una lista
            if isinstance(file_dati, list):
                for entry in file_dati:
                    entry["num_thread"] = num_thread  # Aggiungi il numero di thread ai dati
                    dati.append(entry)
            else:
                file_dati["num_thread"] = num_thread  # Aggiungi il numero di thread ai dati
                dati.append(file_dati)

# Organizza i dati per matrice
matrici = {}
for dato in dati:
    nome = dato["nameMatrix"]
    if nome not in matrici:
        matrici[nome] = []
    matrici[nome].append((dato["num_thread"], dato["megaFlops"], dato["nonzeri"]))

# Ordina i dati per numero di thread
for nome in matrici:
    matrici[nome].sort(key=lambda x: x[0])

# Raggruppa le matrici in base al numero di nonzeri
raggruppamenti = {f"<{soglie_nonzeri[0]}": []}
for i in range(len(soglie_nonzeri) - 1):
    raggruppamenti[f"{soglie_nonzeri[i]}-{soglie_nonzeri[i + 1]}"] = []
raggruppamenti[f">={soglie_nonzeri[-1]}"] = []

for nome, valori in matrici.items():
    nonzeri = valori[0][2]  # I nonzeri sono uguali per tutti i campioni della stessa matrice
    if nonzeri < soglie_nonzeri[0]:
        raggruppamenti[f"<{soglie_nonzeri[0]}"].append((nome, valori))
    elif nonzeri >= soglie_nonzeri[-1]:
        raggruppamenti[f">={soglie_nonzeri[-1]}"].append((nome, valori))
    else:
        for i in range(len(soglie_nonzeri) - 1):
            if soglie_nonzeri[i] <= nonzeri < soglie_nonzeri[i + 1]:
                raggruppamenti[f"{soglie_nonzeri[i]}-{soglie_nonzeri[i + 1]}"].append((nome, valori))
                break

# Crea il grafico con i gruppi di matrici
plt.figure(figsize=(10, 6))

# Per ogni gruppo di matrici, tracciamo l'andamento delle prestazioni
for gruppo, matrici_gruppo in raggruppamenti.items():
    if not matrici_gruppo:
        continue  # Salta i gruppi vuoti

    # Raggruppa i dati per numero di thread
    thread_values = sorted(set([v[0] for nome, valori in matrici_gruppo for v in valori]))
    average_performance = {t: [] for t in thread_values}
    
    for num_thread in thread_values:
        # Per ogni numero di thread, calcola la media delle prestazioni (MegaFlops) per quel gruppo
        performance = []
        for nome, valori in matrici_gruppo:
            for v in valori:
                if v[0] == num_thread:
                    performance.append(v[1])  # Aggiungi MegaFlops per quel numero di thread
        average_performance[num_thread] = np.mean(performance) if performance else 0

    # Traccia la linea per il gruppo
    plt.plot(list(average_performance.keys()), list(average_performance.values()), label=f"Nonzeri {gruppo}")

# Impostazioni del grafico
plt.title("Prestazioni del Calcolo Parallelo vs. Numero di Thread per Gruppi di Matrici")
plt.xlabel("Numero di Thread")
plt.ylabel("Prestazioni (MegaFlops)")

# Posiziona la legenda fuori dal grafico
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True)
plt.tight_layout()

# Mostra il grafico
plt.show()

