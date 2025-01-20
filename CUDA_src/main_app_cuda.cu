#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

#include <helper_cuda.h>

#include "../CUDA_libs/csrOperations.h"
#include "../libs/data_structure.h"
#include "../libs/matrixLists.h"
#include "../libs/mmio.h"
#include "../libs/costants.h"
#include "../CUDA_libs/hllOperations.h"

#include <unistd.h>

#ifdef USER_PIERFRANCESCO
#include "../../cJSON/cJSON.h"
const char *base_path = "../../matrix/";
#elif defined(USER_ANDREA)
#include "../../cJSON/include/cjson/cJSON.h"
const char *base_path = "../../matrix/";
#endif


/* Funzione per controllare e creare directory */
void ensure_directory_exists(const char *path) {
    struct stat st;

    if (stat(path, &st) != 0) {
        // La directory non esiste, quindi viene creata
        if (mkdir(path, 0755) != 0) {
            perror("Errore nella creazione della directory");
            exit(EXIT_FAILURE);
        }
    } else if (!S_ISDIR(st.st_mode)) {
        fprintf(stderr, "Il percorso specificato non è una directory: %s\n", path);
        exit(EXIT_FAILURE);
    }
}

/* Funzione per il preprocessamento delle matrici in input da file */
void preprocess_matrix(matrixData *matrix_data, int i) {
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s%s", base_path, matrix_names[i]);

    FILE *f = fopen(full_path, "r");
    if (!f) {
        perror("Errore nell'apertura del file della matrice");
        exit(EXIT_FAILURE);
    }

    /* Lettura dell'intestazione (banner) del file Matrix Market */
    if (mm_read_banner(f, &matrix_data->matcode) != 0) {
        fprintf(stderr, "Errore nella lettura del banner Matrix Market.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    /* Verifica del formato della matrice */
    if (!mm_is_matrix(matrix_data->matcode) || !mm_is_coordinate(matrix_data->matcode)) {
        fprintf(stderr, "Il file non è in formato matrice sparsa a coordinate.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    /* Lettura dei parametri dimensionali della matrice */
    int M, N, nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        fprintf(stderr, "Errore nella lettura delle dimensioni della matrice.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    matrix_data->row_indices = static_cast<int *>(malloc(nz * sizeof(int)));
    matrix_data->col_indices = static_cast<int *>(malloc(nz * sizeof(int)));
    matrix_data->values = static_cast<double *>(malloc(nz * sizeof(double)));
    matrix_data->M = M;
    matrix_data->N = N;
    matrix_data->nz = nz;

    if (matrix_data->row_indices == nullptr || matrix_data->col_indices == nullptr || matrix_data->values == nullptr || matrix_data->M == 0 || matrix_data->N == 0 || matrix_data->nz == 0) {
        perror("Errore nell'allocazione della memoria.");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    /* contiene preprocessamento matrice pattern */
    for (int j = 0; j < nz; j++) {
        int result;
        double value = 1.0; // Valore predefinito per matrici "pattern"

        if (mm_is_pattern(matrix_data->matcode)) {   // Preprocessamento matrice in formato pattern
            result = fscanf(f, "%d %d", &matrix_data->row_indices[j], &matrix_data->col_indices[j]);
        } else {
            result = fscanf(f, "%d %d %lf", &matrix_data->row_indices[j], &matrix_data->col_indices[j], &value);
        }

        if (result != (mm_is_pattern(matrix_data->matcode) ? 2 : 3)) {
            fprintf(stderr, "Errore nella lettura degli elementi della matrice.\n");
            free(matrix_data->row_indices);
            free(matrix_data->col_indices);
            free(matrix_data->values);
            fclose(f);
            exit(EXIT_FAILURE);
        }

        matrix_data->row_indices[j]--; // Converti a indice 0-based
        matrix_data->col_indices[j]--; // Converti a indice 0-based
        matrix_data->values[j] = value;
    }

    /* Preprocessamento matrice simmetrica */
    if (mm_is_symmetric(matrix_data->matcode)) {
        int extra_nz = 0;
        for (int j = 0; j < nz; j++) {
            if (matrix_data->row_indices[j] != matrix_data->col_indices[j]) {
                extra_nz++;
            }
        }

        // Estensione degli array con il numero di non zeri da aggiungere
        matrix_data->row_indices = static_cast<int *>(realloc(matrix_data->row_indices, (nz + extra_nz) * sizeof(int)));
        matrix_data->col_indices = static_cast<int *>(realloc(matrix_data->col_indices, (nz + extra_nz) * sizeof(int)));
        matrix_data->values = static_cast<double *>(realloc(matrix_data->values, (nz + extra_nz) * sizeof(double)));

        if (matrix_data->row_indices == nullptr || matrix_data->col_indices == nullptr || matrix_data->values == nullptr) {
            perror("Errore nell'allocazione della memoria.");
            fclose(f);
            exit(EXIT_FAILURE);
        }

        // Aggiunta degli elementi simmetrici
        int index = nz;
        for (int j = 0; j < nz; j++) {
            if (matrix_data->row_indices[j] != matrix_data->col_indices[j]) {
                matrix_data->row_indices[index] = matrix_data->col_indices[j];
                matrix_data->col_indices[index] = matrix_data->row_indices[j];
                matrix_data->values[index] = matrix_data->values[j];
                index++;
            }
        }
        matrix_data->nz += extra_nz;
    }
    fclose(f);
}

/* Funzione per aggiungere i risultati in un array JSON */
void add_performance_to_array(const char *nameMatrix,
                              matrixData *matrix_data, double *x,
                              cJSON *matrix_array,
                              matrixPerformance (*calculation_function)(matrixData *, double *)) {
    struct matrixPerformance matrixPerformance{};

    // Esegui il calcolo
    matrixPerformance = calculation_function(matrix_data, x);

    // Copia il nome della matrice
    strncpy(matrixPerformance.nameMatrix, nameMatrix, sizeof(matrixPerformance.nameMatrix) - 1);
    matrixPerformance.nameMatrix[sizeof(matrixPerformance.nameMatrix) - 1] = '\0';

    // Creazione dell'oggetto JSON per l'iterazione corrente
    cJSON *performance_obj = cJSON_CreateObject();
    cJSON_AddStringToObject(performance_obj, "nameMatrix", matrixPerformance.nameMatrix);
    cJSON_AddNumberToObject(performance_obj, "seconds", matrixPerformance.seconds);
    cJSON_AddNumberToObject(performance_obj, "nonzeros", matrix_data->nz);
    cJSON_AddNumberToObject(performance_obj, "flops", 0);
    cJSON_AddNumberToObject(performance_obj, "gigaFlops", 0);

    cJSON_AddNumberToObject(performance_obj, "righe",matrix_data->M);
    cJSON_AddNumberToObject(performance_obj, "colonne", matrix_data->N);

    // Aggiungi l'oggetto all'array JSON
    cJSON_AddItemToArray(matrix_array, performance_obj);
}

/* Funzione per scrivere l'array JSON in un file */
void write_json_to_file(const char *file_path, cJSON *json_array) {
    FILE *file = fopen(file_path, "w");
    if (!file) {
        perror("Errore nell'apertura del file JSON");
        return;
    }

    // Converti l'array JSON in stringa e scrivilo nel file
    char *json_string = cJSON_Print(json_array);
    fprintf(file, "%s\n", json_string);

    free(json_string);
    fclose(file);
}

// Funzione per calcolare la media dei seconds e generare il JSON finale
void calculatePerformance(const char *input_file_path, const char *output_file_path) {
    FILE *input_file = fopen(input_file_path, "r");
    if (input_file == nullptr) {
        perror("Errore nell'apertura del file di input");
        exit(EXIT_FAILURE);
    }

    // Leggi tutto il contenuto del file
    fseek(input_file, 0, SEEK_END);
    long file_size = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);

    char *file_content = static_cast<char *>(malloc(file_size + 1));
    if (file_content == nullptr) {
        perror("Errore nell'allocazione della memoria per il contenuto del file");
        fclose(input_file);
        exit(EXIT_FAILURE);
    }

    fread(file_content, 1, file_size, input_file);
    file_content[file_size] = '\0'; // Assicurati che il contenuto sia terminato da un carattere nullo
    fclose(input_file);

    // Parse del contenuto come un array JSON
    cJSON *json_array = cJSON_Parse(file_content);
    free(file_content);

    if (json_array == nullptr || !cJSON_IsArray(json_array)) {
        fprintf(stderr, "Errore nel parsing del JSON o il file non è un array JSON valido.\n");
        cJSON_Delete(json_array);
        exit(EXIT_FAILURE);
    }

    auto *matrix_results = static_cast<MatrixPerformanceResult *>(malloc(1000 * sizeof(MatrixPerformanceResult))); // Buffer iniziale
    int matrix_result_count = 0;
    int nz = 0;
    int righe,colonne;

    cJSON *item;
    cJSON_ArrayForEach(item, json_array) {
        const char *nameMatrix = cJSON_GetObjectItem(item, "nameMatrix")->valuestring;
        double seconds = cJSON_GetObjectItem(item, "seconds")->valuedouble;
        nz = cJSON_GetObjectItem(item, "nonzeros")->valueint;  // Debug: verifica i valori letti
        righe = cJSON_GetObjectItem(item, "righe")->valueint;
        colonne = cJSON_GetObjectItem(item, "colonne")->valueint;

        // Cerca se il nameMatrix esiste già
        int found = 0;
        for (int i = 0; i < matrix_result_count; i++) {
            if (strcmp(matrix_results[i].nameMatrix, nameMatrix) == 0) {
                matrix_results[i].total_seconds += seconds;
                matrix_results[i].count++;
                found = 1;
                break;
            }
        }

        // Se non trovato, aggiungi un nuovo entry
        if (!found) {
            strncpy(matrix_results[matrix_result_count].nameMatrix, nameMatrix, sizeof(matrix_results[matrix_result_count].nameMatrix) - 1);
            matrix_results[matrix_result_count].nameMatrix[sizeof(matrix_results[matrix_result_count].nameMatrix) - 1] = '\0';
            matrix_results[matrix_result_count].total_seconds = seconds;
            matrix_results[matrix_result_count].count = 1;
            matrix_result_count++;
        }
    }

    cJSON_Delete(json_array);

    // Creazione del file di output
    cJSON *output_array = cJSON_CreateArray();

    for (int i = 0; i < matrix_result_count; i++) {
        // Calcola la media dei seconds
        double average_seconds = matrix_results[i].total_seconds / ITERATION_PER_MATRIX;


        // Calcola FLOPS e gigaFLOPS
        double flops = 2.0 * nz / average_seconds;
        double gigaFlops = flops / 1e9;

        // Creazione dell'oggetto JSON
        cJSON *output_data = cJSON_CreateObject();
        cJSON_AddStringToObject(output_data, "nameMatrix", matrix_results[i].nameMatrix);
        cJSON_AddNumberToObject(output_data, "seconds", average_seconds);
        cJSON_AddNumberToObject(output_data, "flops", flops);
        cJSON_AddNumberToObject(output_data, "gigaFlops", gigaFlops);
        cJSON_AddNumberToObject(output_data, "righe",righe);
        cJSON_AddNumberToObject(output_data, "colonne", colonne);

        // Aggiungi l'oggetto all'array JSON
        cJSON_AddItemToArray(output_array, output_data);
    }

    free(matrix_results);

    // Scrivi l'array JSON nel file di output
    FILE *output_file = fopen(output_file_path, "w");
    if (output_file == nullptr) {
        perror("Errore nell'apertura del file di output");
        cJSON_Delete(output_array);
        exit(EXIT_FAILURE);
    }

    char *json_string = cJSON_Print(output_array);
    fprintf(output_file, "%s\n", json_string);

    free(json_string);
    fclose(output_file);
    cJSON_Delete(output_array);
}

int main() {
    ensure_directory_exists("../result");
    ensure_directory_exists("../result/iteration");
    ensure_directory_exists("../result/final");

    constexpr int num_matrices = std::size(matrix_names);

    // Creazione degli array JSON per questa matrice
    cJSON *cuda_array_csr_serial = cJSON_CreateArray();
    cJSON *cuda_array_csr_parallel_v1 = cJSON_CreateArray();
    cJSON *cuda_array_csr_parallel_v2 = cJSON_CreateArray();
    cJSON *cuda_array_csr_parallel_v3 = cJSON_CreateArray();
    cJSON *cuda_array_hll_parallel_v1= cJSON_CreateArray();
    cJSON *cuda_array_hll_parallel_v2 = cJSON_CreateArray();
    cJSON *cuda_array_hll_parallel_v3 = cJSON_CreateArray();


    for (int i = 0; i < num_matrices; i++) {
        /* Inizializzazione della struct contente le informazioni della matrice su host (CPU) */
        auto *matrix_data_host = static_cast<struct matrixData *>(malloc(sizeof(struct matrixData)));
        if (!matrix_data_host) {
            perror("Errore nell'allocazione della memoria per matrix_data.");
            return EXIT_FAILURE;
        }

        printf("Calcolo su matrice: %s\n", matrix_names[i]);
        preprocess_matrix(matrix_data_host, i);

        /* Inizializzazione del vettore x su host */
        auto *x_h = static_cast<double *>(malloc(matrix_data_host->M * sizeof(double)));
        if (!x_h) {
            perror("Errore nell'allocazione della memoria per il vettore x.");
            free(matrix_data_host);
            return EXIT_FAILURE;
        }
        for (int j = 0; j < matrix_data_host->M; j++) {
            x_h[j] = 1.0;
        }

        for (int j = 0; j < ITERATION_PER_MATRIX; j++) {
            /* Esecuzione seriale su CPU */
            add_performance_to_array(matrix_names[i], matrix_data_host, x_h, cuda_array_csr_serial, serial_csr_cuda);
            // Calcolo parallelo su GPU formato CSR
            add_performance_to_array(matrix_names[i], matrix_data_host, x_h, cuda_array_csr_parallel_v1, parallel_csr_cuda_v1);
            add_performance_to_array(matrix_names[i], matrix_data_host, x_h, cuda_array_csr_parallel_v2, parallel_csr_cuda_v2);
            add_performance_to_array(matrix_names[i], matrix_data_host, x_h, cuda_array_csr_parallel_v3, parallel_csr_cuda_v3);
            // Calcolo parallelo su GPU formato HLL
            add_performance_to_array(matrix_names[i], matrix_data_host, x_h, cuda_array_hll_parallel_v1, parallel_hll_cuda_v1);
            add_performance_to_array(matrix_names[i], matrix_data_host, x_h, cuda_array_hll_parallel_v2, parallel_hll_cuda_v2);
        }

        free(x_h);
        free(matrix_data_host);
    }

    write_json_to_file("../result/iteration/CUDA_serial_CSR.json", cuda_array_csr_serial);
    write_json_to_file("../result/iteration/CUDA_CSR_v1.json", cuda_array_csr_parallel_v1);
    write_json_to_file("../result/iteration/CUDA_CSR_v2.json", cuda_array_csr_parallel_v2);
    write_json_to_file("../result/iteration/CUDA_CSR_v3.json", cuda_array_csr_parallel_v3);
    write_json_to_file("../result/iteration/CUDA_HLL.json", cuda_array_hll_parallel_v1);

    cJSON_Delete(cuda_array_csr_serial);
    cJSON_Delete(cuda_array_csr_parallel_v1);
    cJSON_Delete(cuda_array_csr_parallel_v2);
    cJSON_Delete(cuda_array_csr_parallel_v3);
    cJSON_Delete(cuda_array_hll_parallel_v1);

    calculatePerformance("../result/iteration/CUDA_serial_CSR.json", "../result/final/CUDA_serial_CSR.json");
    calculatePerformance("../result/iteration/CUDA_CSR_v1.json", "../result/final/CUDA_CSR_v1.json");
    calculatePerformance("../result/iteration/CUDA_CSR_v2.json", "../result/final/CUDA_CSR_v2.json");
    calculatePerformance("../result/iteration/CUDA_CSR_v3.json", "../result/final/CUDA_CSR_v3.json");
    calculatePerformance("../result/iteration/CUDA_HLL.json", "../result/final/CUDA_HLL.json");

    return EXIT_SUCCESS;
}