#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../libs/mmio.h"
#include "../libs/matrixLists.h"
#include "../libs/data_structure.h"
#include "../libs/costants.h"
#include "../libs/csrOperations.h"
#include "cjson/cJSON.h"

#ifdef USER_PIERFRANCESCO
const char *base_path = "/home/pierfrancesco/Desktop/matrix/";
#elif defined(USER_ANDREA)
const char *base_path = "/Users/andreaandreoli/matrix/";
#else
const char *base_path = "./matrix/";
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

/* Funzione per il reset della struttura dati utilizzata per la memorizzazione di una matrice */
void clean_matrix_mem(struct matrixData *matrix_data) {
    free(matrix_data->col_indices);
    free(matrix_data->row_indices);
    free(matrix_data->values);
    free(matrix_data->matcode);
    matrix_data->M = 0;
    matrix_data->N = 0;
    matrix_data->nz = 0;
}

/* Funzione per il preprocessamento delle matrici in input da file */
void preprocess_matrix(struct matrixData *matrix_data, int i) {
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s%s", base_path, matrix_names[i]);

    FILE *f = fopen(full_path, "r");
    if (!f) {
        perror("Errore nell'apertura del file");
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

    matrix_data->row_indices = malloc(nz * sizeof(int));
    matrix_data->col_indices = malloc(nz * sizeof(int));
    matrix_data->values = malloc(nz * sizeof(double));
    matrix_data->M = M;
    matrix_data->N = N;
    matrix_data->nz = nz;

    if (matrix_data->row_indices == NULL || matrix_data->col_indices == NULL || matrix_data->values == NULL || matrix_data->M == 0 || matrix_data->N == 0 || matrix_data->nz == 0) {
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
        matrix_data->row_indices = realloc(matrix_data->row_indices, (nz + extra_nz) * sizeof(int));
        matrix_data->col_indices = realloc(matrix_data->col_indices, (nz + extra_nz) * sizeof(int));
        matrix_data->values = realloc(matrix_data->values, (nz + extra_nz) * sizeof(double));

        if (matrix_data->row_indices == NULL || matrix_data->col_indices == NULL || matrix_data->values == NULL) {
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

// Funzione per aggiungere i risultati in un array JSON
void add_performance_to_array(const char *nameMatrix,
                              struct matrixData *matrix_data, double *x,
                              cJSON *matrix_array,
                              struct matrixPerformance (*calculation_function)(struct matrixData *, double *)) {
    struct matrixPerformance matrixPerformance;

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
    /*cJSON_AddNumberToObject(performance_obj, "flops", matrixPerformance.flops);
    cJSON_AddNumberToObject(performance_obj, "megaFlops", matrixPerformance.megaFlops);*/
    cJSON_AddNumberToObject(performance_obj, "flops", 0);
    cJSON_AddNumberToObject(performance_obj, "megaFlops", 0);

    // Aggiungi l'oggetto all'array JSON
    cJSON_AddItemToArray(matrix_array, performance_obj);
}

// Funzione per scrivere l'array JSON in un file
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
    if (input_file == NULL) {
        perror("Errore nell'apertura del file di input");
        exit(EXIT_FAILURE);
    }

    // Leggi tutto il contenuto del file
    fseek(input_file, 0, SEEK_END);
    long file_size = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);

    char *file_content = malloc(file_size + 1);
    if (file_content == NULL) {
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

    if (json_array == NULL || !cJSON_IsArray(json_array)) {
        fprintf(stderr, "Errore nel parsing del JSON o il file non è un array JSON valido.\n");
        cJSON_Delete(json_array);
        exit(EXIT_FAILURE);
    }

    // Dizionario per accumulare i risultati per ogni nameMatrix
    typedef struct {
        char nameMatrix[50];
        double total_seconds;
        int count;
    } MatrixPerformanceResult;

    MatrixPerformanceResult *matrix_results = malloc(1000 * sizeof(MatrixPerformanceResult)); // Buffer iniziale
    int matrix_result_count = 0;
    int nz = 0;

    cJSON *item;
    cJSON_ArrayForEach(item, json_array) {
        const char *nameMatrix = cJSON_GetObjectItem(item, "nameMatrix")->valuestring;
        double seconds = cJSON_GetObjectItem(item, "seconds")->valuedouble;
        nz = cJSON_GetObjectItem(item, "nonzeros")->valueint;  // Debug: verifica i valori letti
        printf("nameMatrix: %s, seconds: %f, nonzeros: %d\n", nameMatrix, seconds, nz);

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

        // Debug: verifica la media calcolata
        printf("Matrix: %s, Total Seconds: %f, Count: %d, Average Seconds: %f\n",
               matrix_results[i].nameMatrix, matrix_results[i].total_seconds, matrix_results[i].count, average_seconds);

        // Calcola FLOPS e MegaFLOPS
        double flops = 2.0 * nz / average_seconds;
        double megaFlops = flops / 1e6;

        // Debug: verifica i valori calcolati
        printf("Matrix: %s, FLOPS: %f, MegaFLOPS: %f\n", matrix_results[i].nameMatrix, flops, megaFlops);

        // Creazione dell'oggetto JSON
        cJSON *output_data = cJSON_CreateObject();
        cJSON_AddStringToObject(output_data, "nameMatrix", matrix_results[i].nameMatrix);
        cJSON_AddNumberToObject(output_data, "seconds", average_seconds);
        cJSON_AddNumberToObject(output_data, "flops", flops);
        cJSON_AddNumberToObject(output_data, "megaFlops", megaFlops);

        // Aggiungi l'oggetto all'array JSON
        cJSON_AddItemToArray(output_array, output_data);
    }

    free(matrix_results);

    // Scrivi l'array JSON nel file di output
    FILE *output_file = fopen(output_file_path, "w");
    if (output_file == NULL) {
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

    const int num_matrices = sizeof(matrix_names) / sizeof(matrix_names[0]);

    // Creazione degli array JSON per questa matrice
    cJSON *serial_array = cJSON_CreateArray();
    cJSON *parallel_array = cJSON_CreateArray();

    for (int i = 0; i < num_matrices; i++) {
        struct matrixData *matrix_data = malloc(sizeof(struct matrixData));
        if (!matrix_data) {
            perror("Errore nell'allocazione della memoria per matrix_data.");
            return EXIT_FAILURE;
        }

        printf("Calcolo su matrice: %s\n", matrix_names[i]);
        preprocess_matrix(matrix_data, i);

        double *x = malloc(matrix_data->N * sizeof(double));
        if (!x) {
            perror("Errore nell'allocazione della memoria per il vettore x.");
            free(matrix_data);
            return EXIT_FAILURE;
        }
        for (int j = 0; j < matrix_data->N; j++) {
            x[j] = 1.0;
        }

        for (int j = 0; j < ITERATION_PER_MATRIX; j++) {
            add_performance_to_array(matrix_names[i], matrix_data, x, serial_array, serial_csr);
            add_performance_to_array(matrix_names[i], matrix_data, x, parallel_array, parallel_csr);
        }

        free(x);
        free(matrix_data);
    }

    write_json_to_file("../result/iteration/serial_CSR.json", serial_array);
    write_json_to_file("../result/iteration/par_OpenMP_CSR.json", parallel_array);

    cJSON_Delete(serial_array);
    cJSON_Delete(parallel_array);

    calculatePerformance("../result/iteration/serial_CSR.json", "../result/final/serial_CSR.json");
    calculatePerformance("../result/iteration/par_OpenMP_CSR.json", "../result/final/par_OpenMP_CSR.json");

    return EXIT_SUCCESS;
}

