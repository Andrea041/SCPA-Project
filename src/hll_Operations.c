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
    int *row_counts = (int *)calloc(M, sizeof(int));
    if (!row_counts) {
        fprintf(stderr, "Errore: Allocazione fallita per row_counts.\n");
        exit(EXIT_FAILURE);
    }

    // Conta il numero di non-zero per ogni riga
    for (int i = 0; i < nz; i++) {
        row_counts[row_indices[i]]++;
    }

    // Trova il massimo numero di non-zero
    int max_nz_per_row = 0;
    for (int i = 0; i < M; i++) {
        if (row_counts[i] > max_nz_per_row) {
            max_nz_per_row = row_counts[i];
        }
    }

    hll_matrix->max_nz_per_row = max_nz_per_row + 1; // Include lo zero aggiuntivo richiesto
    free(row_counts);
}



void distribute_rows_to_threads(int M, HLL_Matrix *hll_matrix, int num_threads, int **start_row, int **end_row, int *valid_threads) {
    *start_row = (int *)malloc(num_threads * sizeof(int));
    *end_row = (int *)malloc(num_threads * sizeof(int));

    if (!(*start_row) || !(*end_row)) {
        fprintf(stderr, "Errore: Allocazione fallita per start_row o end_row.\n");
        exit(EXIT_FAILURE);
    }

    int *non_zero_per_row = (int *)malloc(M * sizeof(int));
    if (!non_zero_per_row) {
        fprintf(stderr, "Errore: Allocazione fallita per non_zero_per_row.\n");
        exit(EXIT_FAILURE);
    }

    // Calcola il numero di non-zero per ogni riga
    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int start_row_block = block_idx * HackSize;
        int end_row_block = (block_idx + 1) * HackSize;
        if (end_row_block > M) end_row_block = M;

        ELLPACK_Block *block = &hll_matrix->blocks[block_idx];

        for (int i = start_row_block; i < end_row_block; i++) {
            int non_zero_count = 0;
            for (int j = 0; j < hll_matrix->max_nz_per_row; j++) {
                int col_idx = block->JA[(i - start_row_block) * hll_matrix->max_nz_per_row + j];
                if (col_idx >= 0) {
                    non_zero_count++;
                }
            }
            non_zero_per_row[i] = non_zero_count;
        }
    }

    // Calcola il numero totale di non-zero e il target per ogni thread
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
    int N = matrix_data->N;
    int nz = matrix_data->nz;
    int *row_indices = matrix_data->row_indices;
    int *col_indices = matrix_data->col_indices;
    double *values = matrix_data->values;

    HLL_Matrix *hll_matrix = (HLL_Matrix *)malloc(sizeof(HLL_Matrix));
    if (!hll_matrix) {
        fprintf(stderr, "Errore: Allocazione fallita per HLL_Matrix.\n");
        exit(EXIT_FAILURE);
    }

    // Calcolo del numero di blocchi
    hll_matrix->num_blocks = (M + HackSize - 1) / HackSize;


    // Calcolo del massimo numero di non-zero per riga
    calculate_max_nz_per_row(M, nz, row_indices, hll_matrix);

    // Allocazione dei blocchi
    hll_matrix->blocks = (ELLPACK_Block *)malloc(hll_matrix->num_blocks * sizeof(ELLPACK_Block));
    if (!hll_matrix->blocks) {
        fprintf(stderr, "Errore: Allocazione fallita per i blocchi ELLPACK.\n");
        free(hll_matrix);
        exit(EXIT_FAILURE);
    }

    // Conversione in formato HLL
    convert_to_hll(M, N, nz, row_indices, col_indices, values, hll_matrix);

    // Stampa della matrice memorizzata in formato HLL
    printf("Formato HLL (ELLPACK):\n");
    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        printf("Blocco %d:\n", block_idx);
        ELLPACK_Block *block = &hll_matrix->blocks[block_idx];
        printf("JA (Indici delle colonne):\n");
        for (int i = 0; i < HackSize * hll_matrix->max_nz_per_row; i++) {
            printf("%d ", block->JA[i]);
        }
        printf("\nAS (Valori non nulli):\n");
        for (int i = 0; i < HackSize * hll_matrix->max_nz_per_row; i++) {
            printf("%.2f ", block->AS[i]);
        }
        printf("\n");
    }

    int num_threads = omp_get_max_threads();
    int *start_row = NULL;
    int *end_row = NULL;
    int valid_threads = 0;

    distribute_rows_to_threads(M, hll_matrix, num_threads, &start_row, &end_row, &valid_threads);


    // Stampa delle righe assegnate a ciascun thread e dei loro non-zeri
    printf("\nDistribuzione delle righe ai thread:\n");
    for (int thread_id = 0; thread_id < valid_threads; thread_id++) {
        int thread_nz = 0; // Numero di non-zeri per il thread corrente
       printf("Thread %d: righe da %d a %d\n", thread_id, start_row[thread_id], end_row[thread_id]);

        for (int i = start_row[thread_id]; i <= end_row[thread_id]; i++) {
            for (int j = 0; j < nz; j++) {
                if (row_indices[j] == i) {
                    thread_nz++;
                }
            }
        }
       printf("Thread %d: numero di non-zeri = %d\n", thread_id, thread_nz);
    }

    double *y = (double *)malloc(M * sizeof(double));
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
