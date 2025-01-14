#include <cstdio>
#include <cstdlib>
#include <helper_timer.h>

#include "../CUDA_libs/csrTool.h"
#include "../libs/data_structure.h"
#include "../CUDA_libs/csrOperations.h"

/* Implementazione del prodotto matrice-vettore seriale su CPU */
matrixPerformance serial_csr_cuda(matrixData *matrix_data_host, double *x_h) {
    int *IRP, *JA;
    double *AS;

    /* Vettore di output del risultato y <- Ax inizializzato su host */
    auto *y_h = static_cast<double *>(malloc(matrix_data_host->M * sizeof(double)));
    if (y_h == nullptr) {
        printf("Errore nell'allocazione della memoria per il vettore di output y\n");
        exit(EXIT_FAILURE);
    }

    /* Conversione in formato CSR */
    convert_to_csr(matrix_data_host->M, matrix_data_host->nz, matrix_data_host->row_indices, matrix_data_host->col_indices, matrix_data_host->values, &IRP, &JA, &AS);

    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);

    timer->start();
    matvec_csr(matrix_data_host->M, IRP, JA, AS, x_h, y_h);
    timer->stop();

    matrixPerformance node{};
    node.seconds = timer->getTime();
    node.flops = 0;
    node.gigaFlops = 0;

    free(y_h);
    free(IRP);
    free(JA);
    free(AS);

    return node;
}
