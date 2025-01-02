#include <ctype.h>
#include <stdio.h>
#include <string.h>

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

/* Funzione per il reset della struttura dati utilizzata per la memorizzazione di una matrice */
void clean_matrix_mem(struct matrixData *matrix_data) {
    free(matrix_data->col_indices);
    free(matrix_data->row_indices);
    free(matrix_data->values);
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
    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "Errore nella lettura del banner Matrix Market.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    /* Verifica del formato della matrice */
    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode)) {
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

        if (mm_is_pattern(matcode)) {   // Preprocessamento matrice in formato pattern
            result = fscanf(f, "%d %d", &matrix_data->row_indices[j], &matrix_data->col_indices[j]);
        } else {
            result = fscanf(f, "%d %d %lf", &matrix_data->row_indices[j], &matrix_data->col_indices[j], &value);
        }

        if (result != (mm_is_pattern(matcode) ? 2 : 3)) {
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
    if (mm_is_symmetric(matcode)) {
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

// Funzione per eseguire il calcolo e aggiungere i risultati al JSON
void add_performance_to_json(const char *nameMatrix, int iteration,
                             struct matrixData *matrix_data, double *x,
                             cJSON *matrix_array,
                             struct matrixPerformance (*calculation_function)(int, int, int *, int *, double *, double *),
                             FILE *output_file) {
    struct matrixPerformance matrixPerformance;

    // Esegui il calcolo utilizzando la funzione fornita
    matrixPerformance = calculation_function(matrix_data->M, matrix_data->nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, x);

    // Copia il nome della matrice nella struttura
    strncpy(matrixPerformance.nameMatrix, nameMatrix, sizeof(matrixPerformance.nameMatrix) - 1);
    matrixPerformance.nameMatrix[sizeof(matrixPerformance.nameMatrix) - 1] = '\0'; // Assicura la terminazione della stringa

    // Creazione dell'oggetto JSON per l'iterazione corrente
    cJSON *performance_obj = cJSON_CreateObject();
    cJSON_AddStringToObject(performance_obj, "nameMatrix", matrixPerformance.nameMatrix);
    cJSON_AddNumberToObject(performance_obj, "iteration", iteration);
    cJSON_AddNumberToObject(performance_obj, "seconds", matrixPerformance.seconds);
    cJSON_AddNumberToObject(performance_obj, "flops", matrixPerformance.flops);
    cJSON_AddNumberToObject(performance_obj, "megaFlops", matrixPerformance.megaFlops);

    // Aggiungi l'oggetto all'array JSON
    cJSON_AddItemToArray(matrix_array, performance_obj);

    // Scrivi direttamente l'oggetto JSON nel file specificato
    char *json_string = cJSON_Print(performance_obj);
    fprintf(output_file, "%s\n", json_string);
    free(json_string);
}

int main() {
    const int num_matrices = sizeof(matrix_names) / sizeof(matrix_names[0]); // Numero di matrici

    struct matrixData *matrix_data = malloc(sizeof(struct matrixData));

    if (matrix_data == NULL) {
        perror("Errore nell'allocazione della memoria per matrix_data.");
        return EXIT_FAILURE;
    } // Controllo malloc matrix_data

    /* Creazione dei file json relativi ai risultati di ciascuna esecuzione di calcolo */
    FILE *file_serial = fopen("../result/serial_CSR.json", "w");
    if (file_serial == NULL) {
        fprintf(stderr, "Errore nell'apertura del file serial_CSR.json\n");
        return EXIT_FAILURE;
    }

    FILE *file_par_MP_CSR = fopen("../result/par_mp_CSR.json", "w");
    if (file_par_MP_CSR == NULL) {
        fprintf(stderr, "Errore nell'apertura del file par_mp_CSR.json\n");
        fclose(file_serial);
        return EXIT_FAILURE;
    }

    double *x = malloc((size_t)matrix_data->N * sizeof(double)); // Vettore per prodotto matrice-vettore

    if (x == NULL) {
        perror("Errore nell'allocazione della memoria per il vettore x.");
        free(matrix_data);
        fclose(file_serial);
        fclose(file_par_MP_CSR);
        return EXIT_FAILURE;
    } // Controllo malloc vettore X

    for (int j = 0; j < matrix_data->N; j++)
        x[j] = 1.0;

    for (int i = 0; i < num_matrices; i++) {
        printf("Calcolo su matrice: %s\n", matrix_names[i]);

        preprocess_matrix(matrix_data, i); // Pre-processamento della matrice

        cJSON *matrix_array = cJSON_CreateArray(); // Array per memorizzare le prestazioni di tutte le iterazioni

        for (int j = 0; j < ITERATION_PER_MATRIX ; j++) {

            // Aggiungi le performance per il calcolo "serial_csr"
            add_performance_to_json(matrix_names[i], j + 1, matrix_data, x, matrix_array, serial_csr, file_serial);

            // Aggiungi le performance per il calcolo "parallel_csr"
            add_performance_to_json(matrix_names[i], j + 1, matrix_data, x, matrix_array, parallel_csr, file_par_MP_CSR);
        }

        /* Qua si fa il calcolo su ogni file per ogni matrice*/

        // Libera la memoria allocata per l'array JSON
        cJSON_Delete(matrix_array);
        clean_matrix_mem(matrix_data); // Reset della memoria della matrice
    }

    // Chiudi i file
    fclose(file_serial);
    fclose(file_par_MP_CSR);

    // Libera la memoria allocata dinamicamente
    free(matrix_data);
    free(x);

    return EXIT_SUCCESS;
}