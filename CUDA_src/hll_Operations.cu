#include "../CUDA_libs/csrTool.h"
#include "../CUDA_libs/csrOperations.h"
#include "../CUDA_libs/cudaCostants.h"
#include "../CUDA_libs/hllTool.h"
#include "../libs/data_structure.h"

#include <helper_cuda.h>
#include <helper_timer.h>


// Funzione principale per calcolare il prodotto parallelo
matrixPerformance parallel_hll_cuda(matrixData *matrix_data_host, double *x_h) {
    double *d_y;
    double *d_x;
    int M = matrix_data_host->M;

    HLL_Matrix *hllMatrixHost = static_cast<HLL_Matrix *> (malloc(sizeof(HLL_Matrix)));
    if (hllMatrixHost == nullptr) {
        printf("Errore nell'allocazione della memoria per il vettore di output y\n");
        exit(EXIT_FAILURE);
    }

    // Calcolo del numero di blocchi
    hllMatrixHost->num_blocks = (M + HackSize - 1) / HackSize;

    // Allocazione dei blocchi
    hllMatrixHost->blocks = (ELLPACK_Block *)malloc((size_t)hllMatrixHost->num_blocks * sizeof(ELLPACK_Block));
    if (!hllMatrixHost->blocks) {
        fprintf(stderr, "Errore: Allocazione fallita per i blocchi ELLPACK.\n");
        free(hllMatrixHost);
        exit(EXIT_FAILURE);
    }

    auto *y_h = static_cast<double *>(malloc(matrix_data_host->M * sizeof(double)));
    if (y_h == nullptr) {
        printf("Errore nell'allocazione della memoria per il vettore di output y\n");
        exit(EXIT_FAILURE);
    }

    // Conversione in formato HLL
    convert_to_hll(matrix_data_host, hllMatrixHost);
    printf("HLL 1\n");
    // Copia della struttura HLL_Matrix e dei suoi blocchi in memoria GPU
    HLL_Matrix *d_hll_matrix;
    cudaMalloc(&d_hll_matrix, sizeof(HLL_Matrix));
    cudaMemcpy(d_hll_matrix, hllMatrixHost, sizeof(HLL_Matrix), cudaMemcpyHostToDevice);
    printf("HLL 2\n");
    ELLPACK_Block *d_blocks;
    cudaMalloc(&d_blocks, hllMatrixHost->num_blocks * sizeof(ELLPACK_Block));
    cudaMemcpy(&d_hll_matrix->blocks, &d_blocks, sizeof(ELLPACK_Block*), cudaMemcpyHostToDevice);
    printf("HLL 3\n");
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        ELLPACK_Block *block = &hllMatrixHost->blocks[i];

        int *d_JA;
        double *d_AS;
        printf("HLL 4\n");
        // Allocazione e copia di JA (indici delle colonne)
        cudaMalloc(&d_JA, block->nz_per_block * sizeof(int));
        cudaMemcpy(d_JA, block->JA, block->nz_per_block * sizeof(int), cudaMemcpyHostToDevice);
        printf("HLL 5\n");
        // Allocazione e copia di AS (valori non nulli)
        cudaMalloc(&d_AS, block->nz_per_block * sizeof(double));
        cudaMemcpy(d_AS, block->AS, block->nz_per_block * sizeof(double), cudaMemcpyHostToDevice);

        // Copia dei puntatori al blocco GPU
        cudaMemcpy(&d_blocks[i].JA, &d_JA, sizeof(int*), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_blocks[i].AS, &d_AS, sizeof(double*), cudaMemcpyHostToDevice);

        // Copia degli altri attributi del blocco
        cudaMemcpy(&d_blocks[i].max_nz_per_row, &block->max_nz_per_row, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_blocks[i].nz_per_block, &block->nz_per_block, sizeof(int), cudaMemcpyHostToDevice);
        printf("HLL &\n");
    }
    printf("HLL 7\n");


    int *start_block = NULL;
    int *end_block = NULL;



    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);

    //const dim3 GRID_DIM((matrix_data_host->M-1+YBD)/YBD); //this way we have the right number of block rows even if m is not multiple of YBD.


    int threads_per_block = 256; // Numero di thread per blocco
    int blocks_per_grid = hllMatrixHost->num_blocks; // Un blocco CUDA per ogni blocco ELLPACK

    checkCudaErrors(cudaMalloc((void **) &d_x, matrix_data_host->N * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_x, x_h, matrix_data_host->M * sizeof(double), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_y, matrix_data_host->N * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_y, y_h, matrix_data_host->M * sizeof(double), cudaMemcpyHostToDevice));

    timer->start();
    gpuMatVec_Hll<<<blocks_per_grid, threads_per_block>>>(hllMatrixHost, d_x, d_y, matrix_data_host->M);
    checkCudaErrors(cudaDeviceSynchronize());   //GPU kernel calls are asynchronous: cudaDeviceSynchronize() is useful to take the actual execution time on the GPU before timer->stop().

    timer->stop();

    checkCudaErrors(cudaMemcpy(y_h, d_y,  matrix_data_host->M * sizeof(double), cudaMemcpyDeviceToHost));

    printf("HLL CUDA\n");

    for (int i = 0; i < matrix_data_host->M; i++) {
        printf("y[%d] = %lf\n", i, y_h[i]);
    }

    matrixPerformance node{};
    node.seconds = timer->getTime();
    node.flops = 0;
    node.gigaFlops = 0;

    free(start_block);
    free(end_block);
    free(hllMatrixHost);

    checkCudaErrors(cudaFree(start_block));
    checkCudaErrors(cudaFree(end_block));
    checkCudaErrors(cudaFree(hllMatrixHost));

    return node;

}