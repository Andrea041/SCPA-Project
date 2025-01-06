#include <time.h>
#include <stdio.h>

#include <stdlib.h>
#include "../libs/hll_ellpack_Tool.h"
#include "../libs/data_structure.h"





struct matrixPerformance parallel_hll(struct matrixData *matrix_data, double *x) {

    int M= matrix_data->M;
    int N= matrix_data->N;
    int nz= matrix_data->nz;
    const int *row_indices;
    const int *col_indices;
    const double *values;

    HLL_Matrix *hll_matrix;
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

    // Conversione al formato HLL
    convert_to_hll(M, N, nz, row_indices, col_indices, values, hll_matrix);

   //distribuzione carico di lavoro


    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    //parallel
    clock_gettime(CLOCK_MONOTONIC, &end);

    const double time_spent = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    node.seconds = time_spent;


    free(y);

    return node;
}