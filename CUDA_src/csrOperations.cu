#include <cstdio>
#include <cstdlib>
#include <helper_timer.h>
#include <cmath>
#include <algorithm>

#include "../CUDA_libs/csrTool.h"
#include "../libs/data_structure.h"
#include "../CUDA_libs/csrOperations.h"
#include "../CUDA_libs/cudaCostants.h"

#include <helper_cuda.h>

double *y_CPU = nullptr;

/* Funzione per verificare la differenza relativa tra il vettore calcola con GPU e quello con CPU */
void checkDifferences(double *y_h, int matrix_row) {
    double relativeDiff = 0.0f;
    double diff = 0.0f;
    double maxAbs;
    double toleranceRel = 1e-6;  // Tolleranza relativa
    double absTolerance = 1e-7;  // Tolleranza per differenze assolute

    for (int i = 0; i < matrix_row; i++) {
        maxAbs = std::max(std::abs(y_CPU[i]), std::abs(y_h[i]));

        // Se entrambi i valori sono molto piccoli, usiamo una tolleranza relativa
        if (maxAbs < toleranceRel) {
            maxAbs = toleranceRel;  // Imposta un valore minimo per maxAbs
        }

        // Calcolo della differenza assoluta
        double currentDiff = std::abs(y_CPU[i] - y_h[i]);

        // Se la differenza assoluta è sufficientemente piccola, consideriamo i numeri uguali
        if (currentDiff <= absTolerance) {
            relativeDiff = 0.0;
            diff = 0.0;
        } else {
            // Calcoliamo la differenza relativa
            relativeDiff = std::max(relativeDiff, currentDiff / maxAbs);
            diff = std::max(diff, currentDiff);
        }

        /* Si garantisce un errore massimo di precisione nell'ordine di e-7
         * una differenza dell'ordine di 1e-7 è generalmente considerata accettabile per la maggior parte dei calcoli numerici
         * su GPU, quindi non c'è motivo di preoccuparsi a meno che tu non abbia requisiti di precisione estremamente elevata.
         */
        if (relativeDiff > toleranceRel)
            printf("Errore: Il valore di y[%d] calcolato (%.10f) non corrisponde al valore calcolato con CPU (%.10f).\n", i, y_h[i], y_CPU[i]);
    }
}

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

    y_CPU = static_cast<double *>(malloc(matrix_data_host->M * sizeof(double)));
    memcpy(y_CPU, y_h, matrix_data_host->M * sizeof(double));

    matrixPerformance node{};
    node.seconds = timer->getTime()/1000.0f;
    node.flops = 0;
    node.gigaFlops = 0;

    //printf("Time taken by CPU: %f\n", timer->getTime() / 1000.0f);

    free(y_h);
    free(IRP);
    free(JA);
    free(AS);

    return node;
}

/* Implementazione del prodotto matrice-vettore seriale su GPU - v1 */
matrixPerformance parallel_csr_cuda_v1(matrixData *matrix_data_host, double *x_h) {
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
    checkCudaErrors(cudaMemset(d_y, 0, matrix_data_host->M * sizeof(double)));

    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);
    /* In questo modo ciascun blocco potrà processare una riga in parallelo */
    const dim3 GRID_DIM(matrix_data_host->M);

    timer->start();
    gpuMatVec_csr<<<GRID_DIM, BLOCK_DIM>>>(d_IRP, d_JA, d_AS, d_x, d_y, matrix_data_host->M);
    checkCudaErrors(cudaDeviceSynchronize());   //GPU kernel calls are asynchronous: cudaDeviceSynchronize() is useful to take the actual execution time on the GPU before timer->stop().
    timer->stop();

    checkCudaErrors(cudaMemcpy(y_h, d_y, matrix_data_host->M * sizeof(double), cudaMemcpyDeviceToHost));

    /*for (int i = 0; i < matrix_data_host->M; i++) {
        printf("y_h[%d] = %lf\n", i, y_h[i]);
    }*/

    /* Con questa funzione controlliamo se il vettore y ottenuto da GPU è uguale a quello su CPU */
    checkDifferences(y_h , matrix_data_host->M);

    matrixPerformance node{};
    node.seconds = timer->getTime() / 1000.0f;
    node.flops = 0;
    node.gigaFlops = 0;

    printf("Time taken by GPU: %f\n", timer->getTime() / 1000.0f);


    free(y_h);
    free(h_IRP);
    free(h_JA);
    free(h_AS);

    checkCudaErrors(cudaFree(d_IRP));
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_AS));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_x));

    return node;
}

/* Implementazione del prodotto matrice-vettore seriale su GPU - v2 */
matrixPerformance parallel_csr_cuda_v2(matrixData *matrix_data_host, double *x_h) {
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
    checkCudaErrors(cudaMemset(d_y, 0, matrix_data_host->M * sizeof(double)));

    /* In questo modo ciascun blocco potrà processare una riga in parallelo */
    const dim3 GRID_DIM(matrix_data_host->M);

    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);

    timer->start();
    gpuMatVec_csr_sm_seq<<<GRID_DIM, BLOCK_DIM, SHARED_MEM_SIZE>>>(d_IRP, d_JA, d_AS, d_x, d_y, matrix_data_host->M);
    checkCudaErrors(cudaDeviceSynchronize());   //GPU kernel calls are asynchronous: cudaDeviceSynchronize() is useful to take the actual execution time on the GPU before timer->stop().
    timer->stop();

    checkCudaErrors(cudaMemcpy(y_h, d_y, matrix_data_host->M * sizeof(double), cudaMemcpyDeviceToHost));

    /* Con questa funzione controlliamo se il vettore y ottenuto da GPU è uguale a quello su CPU */
    checkDifferences(y_h , matrix_data_host->M);

    matrixPerformance node{};
    node.seconds = timer->getTime() / 1000.0f;
    node.flops = 0;
    node.gigaFlops = 0;

    //printf("Time taken by GPU: %f\n", timer->getTime() / 1000.0f);

    free(y_h);
    free(h_IRP);
    free(h_JA);
    free(h_AS);

    checkCudaErrors(cudaFree(d_IRP));
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_AS));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_x));

    return node;
}

/* Implementazione del prodotto matrice-vettore seriale su GPU - v3 */
matrixPerformance parallel_csr_cuda_v3(matrixData *matrix_data_host, double *x_h) {
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
    checkCudaErrors(cudaMemset(d_y, 0, matrix_data_host->M * sizeof(double)));

    /* In questo modo ciascun blocco potrà processare una riga in parallelo */
    const dim3 GRID_DIM(matrix_data_host->M);

    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);

    timer->start();
    gpuMatVec_csr_sm_par<<<GRID_DIM, BLOCK_DIM, SHARED_MEM_SIZE>>>(d_IRP, d_JA, d_AS, d_x, d_y, matrix_data_host->M);
    checkCudaErrors(cudaDeviceSynchronize());   //GPU kernel calls are asynchronous: cudaDeviceSynchronize() is useful to take the actual execution time on the GPU before timer->stop().
    timer->stop();

    checkCudaErrors(cudaMemcpy(y_h, d_y, matrix_data_host->M * sizeof(double), cudaMemcpyDeviceToHost));


    /* Con questa funzione controlliamo se il vettore y ottenuto da GPU è uguale a quello su CPU */
    checkDifferences(y_h , matrix_data_host->M);

    matrixPerformance node{};
    node.seconds = timer->getTime() / 1000.0f;
    node.flops = 0;
    node.gigaFlops = 0;

    //printf("Time taken by GPU: %f\n", timer->getTime() / 1000.0f);


    free(y_h);
    free(h_IRP);
    free(h_JA);
    free(h_AS);

    checkCudaErrors(cudaFree(d_IRP));
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_AS));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_x));

    return node;
}