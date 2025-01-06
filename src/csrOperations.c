#include <time.h>
#include <stdio.h>

#include "../libs/csrOperations.h"

#include <omp.h>
#include <stdlib.h>
#include <tgmath.h>

#include "../libs/csrTool.h"
#include "../libs/data_structure.h"

/* Funzione per svolgere il prodotto matrice-vettore, con memorizzazione CSR della matrice, in modo serializzato */
struct matrixPerformance serial_csr(struct matrixData *matrix_data, double *x) {
    int *IRP, *JA;
    double *AS;

    /* Vettore di output del risultato y <- Ax */
    double *y = malloc((size_t)matrix_data->M * sizeof(double));
    if (y == NULL) {
        printf("Errore nell'allocazione della memoria per il vettore di output y\n");
        exit(EXIT_FAILURE);
    }

    /* Conversione in formato CSR */
    convert_to_csr(matrix_data->M, matrix_data->nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, &IRP, &JA, &AS);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matvec_csr(matrix_data->M, IRP, JA, AS, x, y);
    clock_gettime(CLOCK_MONOTONIC, &end);

    const double time_spent = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

    struct matrixPerformance node;
    node.seconds = time_spent;
    node.flops = 0;
    node.megaFlops = 0;

    free(y);
    free(IRP);
    free(JA);
    free(AS);

    return node;
}

void compute_thread_row_partition(int M, int nz, int num_threads, int *IRP, int **start_row, int **end_row) {
    *start_row = malloc((size_t)num_threads * sizeof(int));
    *end_row = malloc((size_t)num_threads * sizeof(int));

    if (*start_row == NULL || *end_row == NULL) {
        printf("Errore nell'allocazione della memoria per la partizione delle righe\n");
        exit(EXIT_FAILURE);
    }

    for (int t = 0; t < num_threads; t++) {
        (*start_row)[t] = -1;
        (*end_row)[t] = -1;
    }

    int nnz_per_thread = nz / num_threads;
    int current_nnz = 0, current_thread = 0;

    for (int i = 0; i < M; i++) {
        if (current_nnz >= current_thread * nnz_per_thread && (*start_row)[current_thread] == -1) {
            (*start_row)[current_thread] = i;
        }

        if (current_nnz >= (current_thread + 1) * nnz_per_thread || i == M - 1) {
            (*end_row)[current_thread] = i + 1; // Include l'ultima riga
            current_thread++;
            if (current_thread >= num_threads) break;
        }

        current_nnz += IRP[i + 1] - IRP[i];
    }
}

struct matrixPerformance parallel_csr(struct matrixData *matrix_data, double *x) {
    int *IRP, *JA;
    double *AS;

    // Vettore di output del risultato y <- Ax
    double *y = malloc((size_t)matrix_data->M * sizeof(double));
    if (y == NULL) {
        printf("Errore nell'allocazione della memoria per il vettore di output y\n");
        exit(EXIT_FAILURE);
    }

    struct matrixPerformance node;
    node.seconds = 0.0;
    node.flops = 0.0;
    node.megaFlops = 0.0;

    convert_to_csr(matrix_data->M, matrix_data->nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, &IRP, &JA, &AS);

    int num_threads = omp_get_max_threads();
    int *start_row, *end_row;

    compute_thread_row_partition(matrix_data->M, matrix_data->nz, num_threads, IRP, &start_row, &end_row);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matvec_csr_openMP(IRP, JA, AS, x, y, start_row, end_row, num_threads, matrix_data->nz, matrix_data->M);
    clock_gettime(CLOCK_MONOTONIC, &end);

    const double time_spent = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    node.seconds = time_spent;

    free(start_row);
    free(end_row);
    free(y);
    free(IRP);
    free(JA);
    free(AS);

    return node;
}