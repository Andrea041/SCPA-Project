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


/* Funzione per stampare i risultati dalla lista */
void print_results(struct matrixPerformanceAverage *results) {
    printf("Lista dei risultati:\n");
    struct matrixPerformanceAverage *current = results;
    while (current != NULL) {
        printf("----------------------\n");
        printf("Matrix name: %s\n", current->nameMatrix);
        printf("Mean execution time: %f\n", current->avarangeSeconds);
        printf("FLOPS: %f\n", current->avarangeFlops);
        printf("MFLOPS: %f\n", current->avarangeMegaFlops);
    }
}

void create_files_header(char *matrix_name, FILE *fileName1, FILE *fileName2, FILE *fileName3) {
    cJSON *matrix = cJSON_CreateObject();
    cJSON_AddStringToObject(matrix, "Name", matrix_name);

    char *json_string = cJSON_Print(matrix);
    fprintf(fileName1, "%s\n", json_string);
    fprintf(fileName2, "%s\n", json_string);
    fprintf(fileName3, "%s\n", json_string);

    cJSON_Delete(matrix);
    free(json_string);
}

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


int main() {
    const int num_matrices = sizeof(matrix_names) / sizeof(matrix_names[0]); // Numero di matrici

    struct matrixData *matrix_data = malloc(sizeof(struct matrixData));

    if (matrix_data == NULL) {
        perror("Errore nell'allocazione della memoria per matrix_data.");
        return EXIT_FAILURE;
    } //controllo malloc matrix_data

    /* Creazione dei file json relativi ai risultati di ciascuna esecuzione di calcolo */
    FILE *file_serial = fopen("../result/serial_CSR.json", "w");
    if (file_serial == NULL) {
        fprintf(stderr, "Errore nell'apertura del file\n");
        return EXIT_FAILURE;
    }

    FILE *file_par_MP_CSR = fopen("../result/par_mp_CSR.json", "w");
    if (file_par_MP_CSR == NULL) {
        fprintf(stderr, "Errore nell'apertura del file\n");
        return EXIT_FAILURE;
    }

    FILE *file_par_MP_HLL = fopen("../result/par_mp_HLL.json", "w");
    if (file_par_MP_HLL == NULL) {
        fprintf(stderr, "Errore nell'apertura del file\n");
        return EXIT_FAILURE;
    }


    double *x = malloc((size_t)matrix_data->N * sizeof(double)); //vettore per prodotto matice vettore

    if (x == NULL) {
        perror("Errore nell'allocazione della memoria per il vettore x.");
        free(matrix_data);
        return EXIT_FAILURE;
    } //controllo malloc vettore X

    for (int j = 0; j < matrix_data->N; j++)
        x[j] = 1.0;

    for (int i = 0; i < num_matrices; i++) {
        printf("Calcolo su matrice: %s\n", matrix_names[i]);

        preprocess_matrix(matrix_data, i); // pre-processamento della matrice

        cJSON *matrix = cJSON_CreateObject();

        /*
         *  iterazione su ogni matrice
         *          ogni matrice calcola per # ITERATION_PER_MATRIX un tipo specifico di calcolo
         *              ad ogni sotto iterazione , mi aggiungo alla lista delle strutture dati delle perfomance matrixPerformance, tutte le emtriche misurate,
         *              le scrivo su un file json , ogni tipo di calcolo ha il suo file json nella cartella Iteration DATA. Quindi per la amtrice successiva quando per il calcolo di tipo A, sarà nel file json del tipo di calcolo "A"
         *      ad ogni iterazione , leggendo i file json per ongi tipo di calcolo, per ogni matrice che ha lo stesso nome faccio la media dei ripettivi valori tra tutte le ITERATION_PER_MATRIX, per rimepire i ripettivi dati della struttura dati matrixPerformanceAvarange che poi verra scritta su un nuovo file json per ogni tipo di calcolo specifico nella cartella FINAL DATA sempre
         */

        //ogni matrice, ha X iterazioni , per ogni tipo di calcolo.
        // un file per ogni tipo di calcolo , cosi che dentro ho tutte le matrici

        create_files_header(matrix_names[i], file_serial, file_par_MP_CSR, file_par_MP_HLL);

        for (int j = 0; j < ITERATION_PER_MATRIX; j++) {
            struct matrixPerformance matrixPerformance;
            matrixPerformance = serial_csr(matrix_data->M, matrix_data->nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, x);

            cJSON *time_spent = cJSON_CreateArray();
            cJSON *flops = cJSON_CreateArray();
            cJSON *mega_flops = cJSON_CreateArray();

            cJSON_AddItemToArray(time_spent, cJSON_CreateNumber(matrixPerformance.seconds));
            cJSON_AddItemToArray(flops, cJSON_CreateNumber(matrixPerformance.flops));
            cJSON_AddItemToArray(mega_flops, cJSON_CreateNumber(matrixPerformance.megaFlops));

            /* Scrivo i risultati contenuti nei vari array */
            if (j == ITERATION_PER_MATRIX - 1) {
                char *json_string = cJSON_Print(matrix);
                fprintf(file_serial, "%s\n", json_string);
                /* Calcolo risultati su file json */

            }
        }
    }

    return EXIT_SUCCESS;
}