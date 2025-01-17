#include <cstdio>
#include <cstdlib>
#include <helper_timer.h>

#include "../CUDA_libs/csrTool.h"
#include "../libs/data_structure.h"
#include "../CUDA_libs/csrOperations.h"
#include "../CUDA_libs/cudaCostants.h"

#include <helper_cuda.h>

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

/* Implementazione del prodotto matrice-vettore seriale su GPU */
matrixPerformance parallel_csr_cuda(matrixData *matrix_data_host, double *x_h) {
    int *h_IRP, *h_JA;
    double *h_AS;

    int *d_IRP, *d_JA;
    double *d_AS;
    double *d_y;
    double *d_x;

    /* Vettore di output del risultato y <- Ax inizializzato su CPU */
    auto *y_h = static_cast<double *>(malloc(matrix_data_host->M * sizeof(double)));
    if (y_h == nullptr) {
        printf("Errore nell'allocazione della memoria per il vettore di output y\n");
        exit(EXIT_FAILURE);
    }

    /* Conversione in formato CSR */
    convert_to_csr(matrix_data_host->M, matrix_data_host->nz, matrix_data_host->row_indices, matrix_data_host->col_indices, matrix_data_host->values, &h_IRP, &h_JA, &h_AS);

    /* Allocazione della memoria su GPU */
    checkCudaErrors(cudaMalloc((void **) &d_IRP, (matrix_data_host->M + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_JA, matrix_data_host->nz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_AS, matrix_data_host->nz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_y, matrix_data_host->M * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_x, matrix_data_host->N * sizeof(double)));

    /* Copia della memoria da CPU a GPU */
    checkCudaErrors(cudaMemcpy(d_IRP, h_IRP, (matrix_data_host->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_JA, h_JA, matrix_data_host->nz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_AS, h_AS, matrix_data_host->nz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, x_h, matrix_data_host->M * sizeof(double), cudaMemcpyHostToDevice));

    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);

    const dim3 GRID_DIM((matrix_data_host->M + YBD - 1) / YBD); //this way we have the right number of block rows even if m is not multiple of YBD.

    timer->start();
    gpuMatVec_csr<<<GRID_DIM, BLOCK_DIM>>>(d_IRP, d_JA, d_AS, d_x, d_y, matrix_data_host->M);
    checkCudaErrors(cudaDeviceSynchronize());   //GPU kernel calls are asynchronous: cudaDeviceSynchronize() is useful to take the actual execution time on the GPU before timer->stop().
    timer->stop();

    checkCudaErrors(cudaMemcpy(y_h, d_y, matrix_data_host->M * sizeof(double), cudaMemcpyDeviceToHost));

    /*for (int i = 0; i < matrix_data_host->M; i++) {
        printf("y[%d] = %lf\n", i, y_h[i]);
    }*/

    matrixPerformance node{};
    node.seconds = timer->getTime();
    node.flops = 0;
    node.gigaFlops = 0;

    free(y_h);
    free(h_IRP);
    free(h_JA);
    free(h_AS);

    checkCudaErrors(cudaFree(d_IRP));
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_AS));

    return node;
}

