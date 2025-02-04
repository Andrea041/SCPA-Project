import json
import matplotlib.pyplot as plt
import os

def load_json_files(file_list):
    """Carica specifici file JSON e restituisce una lista di dati con il nome del file."""
    data = []
    for file in file_list:
        with open(file, 'r') as f:
            content = json.load(f)
            data.append((os.path.basename(file), content))
    return data

def extract_data(json_data):
    """Estrae i dati di interesse dai file JSON."""
    matrix_data = {}
    nonzeros = {}
    for filename, dataset in json_data:
        for entry in dataset:
            name = entry["nameMatrix"]
            megaFlops = entry["megaFlops"]
            nonzero_count = entry["nonzeri"]  # Recupera il numero di nonzeri
            
            if name not in matrix_data:
                matrix_data[name] = {}
                nonzeros[name] = nonzero_count  # Salva il numero di nonzeri
            
            matrix_data[name][filename] = megaFlops
    
    # Ordina le matrici in base al numero di nonzeri
    sorted_matrices = sorted(matrix_data.keys(), key=lambda x: nonzeros[x])
    return matrix_data, sorted_matrices

def plot_histogram(matrix_data, sorted_matrices):
    """Crea un grafico a istogrammi comparando i megaflops, ordinando per numero di nonzeri."""
    file_names = sorted({file for values in matrix_data.values() for file in values})
    values = [[matrix_data[matrix].get(file, 0) for file in file_names] for matrix in sorted_matrices]
    
    plt.figure(figsize=(14, 7))
    bar_width = 0.2
    indices = range(len(sorted_matrices))
    
    for i, file_name in enumerate(file_names):
        plt.bar([x + i * bar_width for x in indices], [v[i] for v in values], width=bar_width, label=file_name)
    
    plt.xticks([x + (bar_width * (len(file_names) / 2)) for x in indices], sorted_matrices, rotation=90, fontsize=12)
    plt.xlabel("Nome Matrice", fontsize=14)
    plt.ylabel("Megaflops", fontsize=14)
    plt.title("Confronto delle prestazioni delle matrici", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_list = ["CUDA_CSR_v1.json", "CUDA_CSR_v2_32x32.json", "CUDA_CSR_v3_32x32.json", "CUDA_serial_CSR.json"]  # Sostituisci con i nomi dei file JSON specifici
    json_data = load_json_files(file_list)
    matrix_data, sorted_matrices = extract_data(json_data)
    plot_histogram(matrix_data, sorted_matrices)