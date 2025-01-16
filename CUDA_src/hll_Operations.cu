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

    // Pulizia della memoria GPU prima delle allocazioni
    cudaDeviceReset();

    HLL_Matrix *hllMatrixHost = static_cast<HLL_Matrix *> (malloc(sizeof(HLL_Matrix)));
    if (hllMatrixHost == nullptr) {
        printf("Errore nell'allocazione della memoria per il vettore di output y\n");
        exit(EXIT_FAILURE);
    }

    // Calcolo del numero di blocchi
    hllMatrixHost->num_blocks = (M + HackSize - 1) / HackSize;
    printf("Numero di blocchi: %d\n", hllMatrixHost->num_blocks);

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
    convert_to_hll_cuda(matrix_data_host, hllMatrixHost);

    printf("HLL 1\n");

    // Stampa della struttura HLL_Matrix
    printf("HLL_Matrix: num_blocks = %d\n", hllMatrixHost->num_blocks);
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        ELLPACK_Block *block = &hllMatrixHost->blocks[i];
        printf("Blocco %d: max_nz_per_row = %d, nz_per_block = %d\n", i, block->max_nz_per_row, block->nz_per_block);
        printf("JA: ");
        for (int j = 0; j < matrix_data_host->M* block->max_nz_per_row; j++) {
            printf("%d ", block->JA[j]);
        }
        printf("\nAS: ");
        for (int j = 0; j < matrix_data_host->M * block->max_nz_per_row; j++) {
            printf("%f ", block->AS[j]);
        }
        printf("\n");
    }

    // Copia della struttura HLL_Matrix in memoria GPU
    HLL_Matrix *d_hll_matrix;
    cudaMalloc(&d_hll_matrix, sizeof(HLL_Matrix));
    cudaMemcpy(d_hll_matrix, hllMatrixHost, sizeof(HLL_Matrix), cudaMemcpyHostToDevice);
    printf("HLL 2\n");

    // Allocazione di memoria GPU per i blocchi
    ELLPACK_Block *d_blocks;
    cudaMalloc(&d_blocks, hllMatrixHost->num_blocks * sizeof(ELLPACK_Block));
    cudaMemcpy(&d_hll_matrix->blocks, &d_blocks, sizeof(ELLPACK_Block *), cudaMemcpyHostToDevice);

    // Iterazione per copiare i blocchi uno per uno
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        ELLPACK_Block *block = &hllMatrixHost->blocks[i];

        int *d_JA;
        double *d_AS;

        // Allocazione e copia di JA
        cudaMalloc(&d_JA, block->nz_per_block * sizeof(int));
        cudaMemcpy(d_JA, block->JA, block->nz_per_block * sizeof(int), cudaMemcpyHostToDevice);

        // Allocazione e copia di AS
        cudaMalloc(&d_AS, block->nz_per_block * sizeof(double));
        cudaMemcpy(d_AS, block->AS, block->nz_per_block * sizeof(double), cudaMemcpyHostToDevice);

        // Imposta i puntatori nel blocco GPU
        ELLPACK_Block d_block;
        d_block.JA = d_JA;
        d_block.AS = d_AS;
        d_block.max_nz_per_row = block->max_nz_per_row;
        d_block.nz_per_block = block->nz_per_block;

        cudaMemcpy(&d_blocks[i], &d_block, sizeof(ELLPACK_Block), cudaMemcpyHostToDevice);
        printf("Blocco %d copiato in GPU\n", i);

        // Debug: Verifica i dati copiati su GPU
        int *debug_JA = (int *)malloc(block->nz_per_block * sizeof(int));
        double *debug_AS = (double *)malloc(block->nz_per_block * sizeof(double));

        cudaMemcpy(debug_JA, d_JA, block->nz_per_block * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(debug_AS, d_AS, block->nz_per_block * sizeof(double), cudaMemcpyDeviceToHost);

        printf("JA (GPU -> CPU): ");
        for (int j = 0; j < block->nz_per_block; j++) {
            printf("%d ", debug_JA[j]);
        }
        printf("\nAS (GPU -> CPU): ");
        for (int j = 0; j < block->nz_per_block; j++) {
            printf("%f ", debug_AS[j]);
        }
        printf("\n");

        free(debug_JA);
        free(debug_AS);
    }


    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);

    checkCudaErrors(cudaMalloc((void **) &d_x, matrix_data_host->M * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_x, x_h, matrix_data_host->M * sizeof(double), cudaMemcpyHostToDevice));
    printf("Copiato x in GPU\n");

    checkCudaErrors(cudaMalloc((void **) &d_y, matrix_data_host->M * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_y, y_h, matrix_data_host->M * sizeof(double), cudaMemcpyHostToDevice));
    printf("Copiato y iniziale in GPU\n");



    const dim3 GRID_DIM((matrix_data_host->M-1+YBD)/YBD);

    timer->start();
    gpuMatVec_Hll<<<GRID_DIM, BLOCK_DIM>>>(hllMatrixHost, d_x, d_y, matrix_data_host->M);
    checkCudaErrors(cudaDeviceSynchronize());

    timer->stop();

    checkCudaErrors(cudaMemcpy(y_h, d_y,  matrix_data_host->M * sizeof(double), cudaMemcpyDeviceToHost));
    printf("Risultato copiato da GPU a CPU:\n");
    for (int i = 0; i < matrix_data_host->M; i++) {
        printf("y[%d] = %lf\n", i, y_h[i]);
    }

    matrixPerformance node{};
    node.seconds = timer->getTime();
    node.flops = 0;
    node.gigaFlops = 0;


    free(hllMatrixHost);

    checkCudaErrors(cudaFree(hllMatrixHost));

    return node;
}
