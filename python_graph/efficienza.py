import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Nuove soglie di nonzeri
soglie_nonzeri = [10000, 100000, 500000, 1e6, 2.5e6, 1e7]
etichette_soglie = [
    "Nonzeri <10000",
    "Nonzeri 10000-100000",
    "Nonzeri 100000-500000",
    "Nonzeri 500000-1000000",
    "Nonzeri 1000000-2500000",
    "Nonzeri 2500000-10000000",
    "Nonzeri >=10000000"
]
colori = ['b', 'orange', 'g', 'r', 'purple', 'brown', 'pink']  # Colori coerenti con la legenda

# Directory contenente i file JSON (aggiorna con il percorso corretto)
dir_json = "."

# Dizionari per memorizzare i dati
dati_serial = {}
dati_parallel = {}
num_nonzeri = {}

# Leggi tutti i file JSON nella directory
for file_name in os.listdir(dir_json):
    if file_name.endswith(".json"):
        try:
            num_thread = int(file_name.split("_")[-1].split(".")[0])  
        except ValueError:
            print(f"Skipping file {file_name} due to invalid thread number.")
            continue  

        with open(os.path.join(dir_json, file_name), "r") as f:
            file_dati = json.load(f)

            for entry in file_dati:
                name_matrix = entry["nameMatrix"]
                nonzeros = entry["nonzeri"]  
                num_nonzeri[name_matrix] = nonzeros

                # Dati seriali
                if "serial" in file_name:
                    if name_matrix not in dati_serial:
                        dati_serial[name_matrix] = {}
                    dati_serial[name_matrix][num_thread] = entry["seconds"]

                # Dati paralleli
                if "par" in file_name and "HLL" in file_name:
                    if name_matrix not in dati_parallel:
                        dati_parallel[name_matrix] = {}
                    dati_parallel[name_matrix][num_thread] = entry["seconds"]

# Calcolare lo efficiency per ogni matrice
matrici_efficiency = {}

for matrix_name in dati_serial:
    if matrix_name in dati_parallel:
        matrici_efficiency[matrix_name] = {}
        for num_thread in dati_parallel[matrix_name]:
            if num_thread in dati_serial[matrix_name]:
                serial_time = dati_serial[matrix_name][num_thread]
                parallel_time = dati_parallel[matrix_name][num_thread]
                if parallel_time > 0:  
                    efficiency = serial_time / (parallel_time * num_thread)
                    matrici_efficiency[matrix_name][num_thread] = efficiency

# Raggruppare gli efficiency medi per soglie di nonzeri
gruppi_efficiency = {s: {} for s in soglie_nonzeri}  

for matrix_name, efficiencys in matrici_efficiency.items():
    nonzeros = num_nonzeri[matrix_name]

    # Trova la soglia di appartenenza della matrice
    for i, soglia in enumerate(soglie_nonzeri):
        if nonzeros <= soglia:
            for num_thread, efficiency in efficiencys.items():
                if num_thread not in gruppi_efficiency[soglia]:
                    gruppi_efficiency[soglia][num_thread] = []
                gruppi_efficiency[soglia][num_thread].append(efficiency)
            break
    else:
        # Se non è stato assegnato a nessuna soglia, va nell'ultimo gruppo (>= 10M)
        soglia = soglie_nonzeri[-1]
        for num_thread, efficiency in efficiencys.items():
            if num_thread not in gruppi_efficiency[soglia]:
                gruppi_efficiency[soglia][num_thread] = []
            gruppi_efficiency[soglia][num_thread].append(efficiency)

# Calcolare lo efficiency medio per ogni intervallo di nonzeri
efficiency_medio = {}

for soglia, efficiency_threads in gruppi_efficiency.items():
    efficiency_medio[soglia] = {}
    for num_thread, valori_efficiency in efficiency_threads.items():
        if valori_efficiency:
            efficiency_medio[soglia][num_thread] = np.mean(valori_efficiency)

# Creare un unico grafico con curve per ogni intervallo
plt.figure(figsize=(12, 7))

for i, (soglia, efficiencys) in enumerate(efficiency_medio.items()):
    if efficiencys:  # Controlla se ci sono dati per questa soglia
        thread_values = sorted(efficiencys.keys())
        efficiency_values = [efficiencys[t] for t in thread_values]

        plt.plot(thread_values, efficiency_values, linestyle='-', 
                 color=colori[i], label=etichette_soglie[i])

# Impostazioni del grafico
plt.title("Efficienza Media vs. Numero di Thread")
plt.xlabel("Numero di Thread")
plt.ylabel("Efficienza Media")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Mostra il grafico
plt.show()
