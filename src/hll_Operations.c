#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Include necessario per memset

#include "../libs/hll_ellpack_Tool.h"
#include "../libs/data_structure.h"
#include "../libs/costants.h"


// Funzione per calcolare il massimo numero di non zero per riga
void calculate_max_nz_per_row(int M, int nz, const int *row_indices, HLL_Matrix *hll_matrix) {
    // Array per contare i non-zero per ogni riga
    int *row_counts = calloc(M, sizeof(int));
    if (!row_counts) {
        fprintf(stderr, "Errore: Allocazione fallita per row_counts.\n");
        exit(EXIT_FAILURE);
    }

    // Conta il numero di non-zero per ogni riga
    for (int i = 0; i < nz; i++) {
        if (row_indices[i] >= M || row_indices[i] < 0) {
            fprintf(stderr, "Errore: Indice di riga fuori dai limiti. row_indices[%d]=%d\n", i, row_indices[i]);
            free(row_counts);
            exit(EXIT_FAILURE);
        }
        row_counts[row_indices[i]]++;
    }

    // Trova il massimo numero di non-zero tra tutte le righe
    int max_nz_per_row = 0;
    for (int i = 0; i < M; i++) {
        if (row_counts[i] > max_nz_per_row) {
            max_nz_per_row = row_counts[i];
        }
    }

    // Debug: Stampa del numero massimo di non-zero
    printf("Numero massimo di non-zero per riga: %d\n", max_nz_per_row);

    // Salva il valore massimo nella struttura HLL_Matrix
    hll_matrix->max_nz_per_row = max_nz_per_row;

    // Libera la memoria temporanea
    free(row_counts);
}




void distribute_rows_to_threads(int M, HLL_Matrix *hll_matrix, int num_threads, int **start_row, int **end_row, int *valid_threads) {
    *start_row = (int *)malloc((size_t)num_threads * sizeof(int));
    *end_row = (int *)malloc((size_t)num_threads * sizeof(int));

    if (!*start_row || !*end_row) {
        fprintf(stderr, "Errore: Allocazione fallita per start_row o end_row.\n");
        exit(EXIT_FAILURE);
    }

    int *non_zero_per_row = malloc(M * sizeof(int));
    if (!non_zero_per_row) {
        fprintf(stderr, "Errore: Allocazione fallita per non_zero_per_row.\n");
        exit(EXIT_FAILURE);
    }
    printf("Blocco 1 completato\n");
    // Calcola il numero di non-zero per ogni riga
    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int start_row_block = block_idx * HackSize;
        int end_row_block = (block_idx + 1) * HackSize;
        if (end_row_block > M) end_row_block = M;

        ELLPACK_Block *block = &hll_matrix->blocks[block_idx];

        for (int i = start_row_block; i < end_row_block; i++) {
            int non_zero_count = 0;

            for (int j = 0; j < hll_matrix->max_nz_per_row; j++) {
                // Calculate the index in the JA array
                int idx = (i - start_row_block) * hll_matrix->max_nz_per_row + j;

                // Ensure the index is within bounds
                if (idx < 0 || idx >= hll_matrix->max_nz_per_row * (end_row_block - start_row_block)) {
                    fprintf(stderr, "Errore: Indice fuori dai limiti. i=%d, j=%d, idx=%d, JA_size=%d\n", i, j, idx, hll_matrix->max_nz_per_row * (end_row_block - start_row_block));
                    exit(EXIT_FAILURE);
                }

                int col_idx = block->JA[idx];
                if (col_idx >= 0) {
                    non_zero_count++;
                }
            }

            non_zero_per_row[i] = non_zero_count;
        }

        //printf("Block %d completed!\n", block_idx);
    }

    printf("Blocco 2 completato\n");
    // Calcola il numero totale di non-zero
    int total_non_zero = 0;
    for (int i = 0; i < M; i++) {
        total_non_zero += non_zero_per_row[i];
    }

    int target_non_zero_per_thread = total_non_zero / num_threads;
    //printf("Total non zero: %d\n", total_non_zero);

    // Suddivisione delle righe ai thread
    int current_non_zero = 0;
    int thread_id = 0;
    (*start_row)[thread_id] = 0;
    printf("Blocco 3 completato\n");
    for (int i = 0; i < M; i++) {
        // Accumula i non-zero per il thread corrente
        current_non_zero += non_zero_per_row[i];

        // Se il carico del thread ha raggiunto il target o siamo all'ultima riga, chiudi il thread
        if ((current_non_zero >= target_non_zero_per_thread && thread_id < num_threads - 1) || i == M - 1) {
            (*end_row)[thread_id] = i; // Chiudi il thread corrente
            thread_id++;

            // Avvia un nuovo thread, se possibile
            if (thread_id < num_threads) {
                (*start_row)[thread_id] = i + 1;
                current_non_zero = 0; // Reset del conteggio per il nuovo thread
            }
        }
    }
    printf("Blocco 4 completato\n");
    // Se il numero di thread è minore del numero richiesto, aggiorna valid_threads
    *valid_threads = thread_id;


    // Libera memoria
    free(non_zero_per_row);
}


// Funzione principale per calcolare il prodotto parallelo
struct matrixPerformance parallel_hll(struct matrixData *matrix_data, double *x) {
    int M = matrix_data->M;
    int N = matrix_data->N;
    int nz = matrix_data->nz;
    int *row_indices = matrix_data->row_indices;
    int *col_indices = matrix_data->col_indices;
    double *values = matrix_data->values;

    HLL_Matrix *hll_matrix = malloc(sizeof(HLL_Matrix));
    if (!hll_matrix) {
        fprintf(stderr, "Errore: Allocazione fallita per HLL_Matrix.\n");
        exit(EXIT_FAILURE);
    }

    // Calcolo del numero di blocchi
    hll_matrix->num_blocks = (M + HackSize - 1) / HackSize;
    printf("Numero di blocchi da utilizzare: %d\n", hll_matrix->num_blocks);

    // Calcolo del massimo numero di non-zero per riga
    calculate_max_nz_per_row(M, nz, row_indices, hll_matrix);

    // Allocazione dei blocchi
    hll_matrix->blocks = (ELLPACK_Block *)malloc((size_t)hll_matrix->num_blocks * sizeof(ELLPACK_Block));
    if (!hll_matrix->blocks) {
        fprintf(stderr, "Errore: Allocazione fallita per i blocchi ELLPACK.\n");
        free(hll_matrix);
        exit(EXIT_FAILURE);
    }

    // Conversione in formato HLL
    convert_to_hll(M, N, nz, row_indices, col_indices, values, hll_matrix);

    // Stampa della matrice memorizzata in formato HLL
    /*printf("Formato HLL (ELLPACK):\n");
    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        printf("Blocco %d:\n", block_idx);
        ELLPACK_Block *block = &hll_matrix->blocks[block_idx];
        printf("JA (Indici delle colonne):\n");
        for (int i = 0; i < matrix_data->M * hll_matrix->max_nz_per_row; i++) {
            printf("%d ", block->JA[i]);
        }
        printf("\nAS (Coefficienti):\n");
        for (int i = 0; i < matrix_data->M * hll_matrix->max_nz_per_row; i++) {
            printf("%.2f ", block->AS[i]);
        }
        printf("\n");
    }*/
    printf("Blocco FFF completato\n");
    int num_threads = omp_get_max_threads();
    int *start_row = NULL;
    int *end_row = NULL;
    int valid_threads = 0;
    printf("Blocco EEE completato\n");
    distribute_rows_to_threads(M, hll_matrix, num_threads, &start_row, &end_row, &valid_threads);
    printf("Blocco HHH completato\n");
    // Stampa delle righe assegnate a ciascun thread e dei loro non-zeri
    // Pre-elaborazione degli intervalli di indici per ciascuna riga
    int *row_start = calloc(M + 1, sizeof(int));
    for (int j = 0; j < nz; j++) {
        row_start[row_indices[j] + 1]++;
    }
    for (int i = 1; i <= M; i++) {
        row_start[i] += row_start[i - 1];
    }
    //printf("\nDistribuzione delle righe ai thread:\n");
    #pragma omp parallel for
    for (int thread_id = 0; thread_id < valid_threads; thread_id++) {
        int thread_nz = 0;
        for (int i = start_row[thread_id]; i <= end_row[thread_id]; i++) {
            thread_nz += row_start[i + 1] - row_start[i]; // Non-zeri in questa riga
        }
        //printf("Thread %d: numero di non-zeri = %d\n", thread_id, thread_nz);
    }
    free(row_start);

    double *y = malloc(M * sizeof(double));
    if (!y) {
        fprintf(stderr, "Errore: Allocazione fallita per il vettore y.\n");
        free(hll_matrix->blocks);
        free(hll_matrix);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
    }
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Calcolo del prodotto matrice-vettore
    matvec_Hll(hll_matrix, x, y, valid_threads, start_row, end_row, N, M);

    clock_gettime(CLOCK_MONOTONIC, &end);

    const double time_spent = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

    struct matrixPerformance performance;
    performance.seconds = time_spent;

    // Libera memoria
    free(y);
    free(start_row);
    free(end_row);
    for (int i = 0; i < hll_matrix->num_blocks; i++) {
        free(hll_matrix->blocks[i].JA);
        free(hll_matrix->blocks[i].AS);
    }
    free(hll_matrix->blocks);
    free(hll_matrix);

    return performance;
}
