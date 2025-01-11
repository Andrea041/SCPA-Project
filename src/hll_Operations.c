#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "../libs/hll_ellpack_Tool.h"
#include "../libs/data_structure.h"
#include "../libs/costants.h"

void distribute_blocks_to_threads(struct matrixData *matrix_data, HLL_Matrix *hll_matrix, int num_threads, int **start_block, int **end_block, int *valid_threads) {
    *start_block = (int *)malloc((size_t)num_threads * sizeof(int));
    *end_block = (int *)malloc((size_t)num_threads * sizeof(int));

    if (!*start_block || !*end_block) {
        fprintf(stderr, "Errore: Allocazione fallita per start_block o end_block.\n");
        exit(EXIT_FAILURE);
    }

    int *non_zero_per_row = calloc(matrix_data->M, sizeof(int));
    if (!non_zero_per_row) {
        fprintf(stderr, "Errore: Allocazione fallita per non_zero_per_row.\n");
        exit(EXIT_FAILURE);
    }

    // Calcola il numero di non-zero per ogni riga
    calculate_max_nz_in_row_in_block(matrix_data, non_zero_per_row);

    // Calcola il numero totale di non-zero
    int total_non_zero = 0;
    for (int i = 0; i < matrix_data->M; i++) {
        total_non_zero += non_zero_per_row[i];
    }

    int non_zero_per_thread = total_non_zero / num_threads;
    int current_non_zero = 0;
    int thread_id = 0;

    (*start_block)[thread_id] = 0; // Primo blocco assegnato al primo thread

    // Assegna i blocchi ai thread
    for (int block_id = 0; block_id < hll_matrix->num_blocks; block_id++) {
        // Aggiunge i non-zero del blocco corrente
        current_non_zero += hll_matrix->blocks[block_id].nz_per_block;

        // Controlla se è necessario passare al prossimo thread
        if (current_non_zero >= non_zero_per_thread && thread_id < num_threads - 1) {
            (*end_block)[thread_id] = block_id; // Assegna il blocco finale per il thread corrente
            thread_id++;
            (*start_block)[thread_id] = block_id + 1; // Inizia il prossimo thread dal blocco successivo
            current_non_zero = 0; // Reset per il prossimo thread
        }
    }

    // Assegna gli ultimi blocchi all'ultimo thread
    if (hll_matrix->num_blocks > 1) {
        (*end_block)[thread_id] = hll_matrix->num_blocks - 1;

        // Aggiorna il numero di thread validi
        *valid_threads = thread_id + 1;
    } else
        *valid_threads = thread_id;

    // Libera memoria temporanea
    free(non_zero_per_row);
}

// Funzione principale per calcolare il prodotto parallelo
struct matrixPerformance parallel_hll(struct matrixData *matrix_data, double *x) {
    int M = matrix_data->M;

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

    int num_threads = omp_get_max_threads();
    int *start_block = NULL;
    int *end_block = NULL;
    int valid_threads = 0;

    /* Distribuzione dei blocchi tra i thread */
    distribute_blocks_to_threads(matrix_data, hll_matrix, num_threads, &start_block, &end_block, &valid_threads);
    printf("HLL Numero di thread: %d\n", num_threads);
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
    matvec_Hll(hll_matrix, x, y, valid_threads, start_block, end_block, matrix_data->M);
    clock_gettime(CLOCK_MONOTONIC, &end);

    const double time_spent = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

    struct matrixPerformance performance;
    performance.seconds = time_spent;
    performance.flops = 0;
    performance.megaFlops = 0;

    // Libera memoria
    free(y);
    free(start_block);
    free(end_block);
    for (int i = 0; i < hll_matrix->num_blocks; i++) {
        free(hll_matrix->blocks[i].JA);
        free(hll_matrix->blocks[i].AS);
    }
    free(hll_matrix->blocks);
    free(hll_matrix);

    return performance;
}
