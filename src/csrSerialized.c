#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "../libs/csrSerialized.h"
#include "../libs/csrTool.h"

void serial_csr(int M, int N, int nz, int *row_indices, int *col_indices, double *values, double *x) {
    int *IRP, *JA;
    double *AS;

    /* Vettore di output del risultato y <- Ax */
    double *y = malloc((size_t)M * sizeof(double));

    convert_to_csr(M, N, nz, row_indices, col_indices, values, &IRP, &JA, &AS);

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    matvec_csr(M, IRP, JA, AS, x, y);
    clock_gettime(CLOCK_MONOTONIC, &end);

    const double time_spent = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;


    const double flops = 2.0 * nz / time_spent;
    const double mflops = flops / 1e6;

    printf("Risultato del prodotto matrice-vettore:\n");
    for (int j = 0; j < M; j++) {
        printf("y[%d] = %.2f\n", j, y[j]);
    }

    printf("Tempo per il prodotto matrice-vettore: %.6f secondi\n", time_spent);
    printf("Performance: %.2f FLOPS\n", flops);
    printf("Performance: %.2f MFLOPS\n", mflops);
}
