#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "../libs/hll_ellpack_Tool.h"
#include "../libs/data_structure.h"
#include "../libs/costants.h"

void printJAandAS(ELLPACK_Block *block, struct matrixData *matrix_data) {
    printf("Vettore JA (Indici delle colonne) per il blocco:\n");

    // Stampa degli indici delle colonne (vettore JA)
    for (int i = 0; i < matrix_data->M; i++) {
        printf("Riga %d: ", i);  // Indice della riga (1-based)
        for (int j = 0; j < block->max_nz_per_row; j++) {
            int col_idx = block->JA[i * block->max_nz_per_row + j];  // Indice della colonna
            printf("%d ", col_idx);  // Indice di colonna (1-based)
        }
        printf("\n");
    }

    printf("\nVettore AS (Valori non nulli) per il blocco:\n");

    // Stampa dei valori non nulli (vettore AS)
    for (int i = 0; i < matrix_data->M; i++) {
        printf("Riga %d: ", i);  // Indice della riga (1-based)
        for (int j = 0; j < block->max_nz_per_row; j++) {
            double value = block->AS[i * block->max_nz_per_row + j];  // Valore non nullo
            printf("%.2f ", value);  // Stampa il valore (fino a 2 decimali)
        }
        printf("\n");
    }
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
    // Calcola il numero di non-zero per ogni riga
    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int max_nz_per_row = hll_matrix->blocks[block_idx].max_nz_per_row;
        int start_row_block = block_idx * HackSize;
        int end_row_block = (block_idx + 1) * HackSize;
        if (end_row_block > M) end_row_block = M;

        ELLPACK_Block *block = &hll_matrix->blocks[block_idx];

        for (int i = start_row_block; i < end_row_block; i++) {
            int non_zero_count = 0;

            for (int j = 0; j < max_nz_per_row; j++) {
                // Calculate the index in the JA array
                const int idx = (i - start_row_block) * max_nz_per_row + j;

                // Ensure the index is within bounds
                if (idx < 0 || idx >= max_nz_per_row * (end_row_block - start_row_block)) {
                    fprintf(stderr, "Errore: Indice fuori dai limiti. i=%d, j=%d, idx=%d, JA_size=%d\n", i, j, idx, max_nz_per_row * (end_row_block - start_row_block));
                    exit(EXIT_FAILURE);
                }

                int col_id = block->JA[idx];
                if (col_id >= 0)
                    non_zero_count++;
            }

            non_zero_per_row[i] = non_zero_count;
        }
    }


    // Calcola il numero totale di non-zero
    int total_non_zero = 0;
    for (int i = 0; i < M; i++) {
        total_non_zero += non_zero_per_row[i];
    }

    int target_non_zero_per_thread = total_non_zero / num_threads;

    // Suddivisione delle righe ai thread
    int current_non_zero = 0;
    int thread_id = 0;
    (*start_row)[thread_id] = 0;

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

    // Se il numero di thread è minore del numero richiesto, aggiorna valid_threads
    *valid_threads = thread_id;


    // Libera memoria
    free(non_zero_per_row);
}


// Funzione principale per calcolare il prodotto parallelo
struct matrixPerformance parallel_hll(struct matrixData *matrix_data, double *x) {
    int M = matrix_data->M;
    int nz = matrix_data->nz;
    int *row_indices = matrix_data->row_indices;

    HLL_Matrix *hll_matrix = malloc(sizeof(HLL_Matrix));
    if (!hll_matrix) {
        fprintf(stderr, "Errore: Allocazione fallita per HLL_Matrix.\n");
        exit(EXIT_FAILURE);
    }

    // Calcolo del numero di blocchi
    hll_matrix->num_blocks = (M + HackSize - 1) / HackSize;

    // Allocazione dei blocchi
    hll_matrix->blocks = (ELLPACK_Block *)malloc((size_t)hll_matrix->num_blocks * sizeof(ELLPACK_Block));
    if (!hll_matrix->blocks) {
        fprintf(stderr, "Errore: Allocazione fallita per i blocchi ELLPACK.\n");
        free(hll_matrix);
        exit(EXIT_FAILURE);
    }

    // Conversione in formato HLL
    convert_to_hll(matrix_data, hll_matrix);
    /*for (int i = 0; i < hll_matrix->num_blocks; i++) {
        printf("\nStampa del blocco %d:\n", i + 1);
        printJAandAS(&hll_matrix->blocks[i], matrix_data);
    }*/

    int num_threads = omp_get_max_threads();
    int *start_row = NULL;
    int *end_row = NULL;
    int valid_threads = 0;

    distribute_rows_to_threads(M, hll_matrix, num_threads, &start_row, &end_row, &valid_threads);

    // Pre-elaborazione degli intervalli di indici per ciascuna riga
    int *row_start = calloc(M + 1, sizeof(int));
    for (int j = 0; j < nz; j++) {
        row_start[row_indices[j] + 1]++;
    }
    for (int i = 1; i <= M; i++) {
        row_start[i] += row_start[i - 1];
    }

    for (int thread_id = 0; thread_id < valid_threads; thread_id++) {
        int thread_nz = 0;
        for (int i = start_row[thread_id]; i <= end_row[thread_id]; i++) {
            thread_nz += row_start[i + 1] - row_start[i]; // Non-zeri in questa riga
        }
    }
    free(row_start);

    double *y = malloc((size_t)M * sizeof(double));
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
    matvec_Hll(hll_matrix, x, y, valid_threads, M);
    clock_gettime(CLOCK_MONOTONIC, &end);

    const double time_spent = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

    struct matrixPerformance performance;
    performance.seconds = time_spent;
    performance.flops = 0;
    performance.megaFlops = 0;

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
