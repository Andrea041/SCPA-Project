#include "../libs/costants.h"
#include "../CUDA_libs/csrOperations.h"
#include "../CUDA_libs/cudaCostants.h"
#include "../CUDA_libs/hllTool.h"
#include "../libs/data_structure.h"

#include <helper_cuda.h>
#include <helper_timer.h>

// Configura la griglia dei blocchi e dei thread
void configure_grid_warp(int M, int sm_count, int *blocks, int *threads) {
  //  printf("Configurazione griglia: M=%d, sm_count=%d\n", M, sm_count);
    int total_threads = ((M + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE; // Allinea a warp
    *threads = WARP_SIZE;  // Ogni blocco ha un warp
    *blocks = (total_threads + *threads - 1) / *threads;
  //  printf("Threads per blocco: %d, Numero di blocchi: %d\n", *threads, *blocks);

    // Assicura che il numero di warp sia multiplo di SM count
    if (*blocks % sm_count != 0) {
        *blocks = (*blocks / sm_count + 1) * sm_count;
    }
}

// Funzione principale per calcolare il prodotto parallelo
matrixPerformance parallel_hll_cuda_v1(matrixData *matrix_data_host, double *x_h) {
    double *d_y;
    double *d_x;
    int M = matrix_data_host->M;

    // Pulizia della memoria GPU prima delle allocazioni
    cudaDeviceReset();
    // Controllo memoria iniziale
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

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
    convert_to_hll_cuda(matrix_data_host, hllMatrixHost);


        // Copia della struttura HLL_Matrix in memoria GPU
    HLL_Matrix *d_hll_matrix;
    checkCudaErrors(cudaMalloc(&d_hll_matrix, sizeof(HLL_Matrix)));
    checkCudaErrors(cudaMemcpy(d_hll_matrix, hllMatrixHost, sizeof(HLL_Matrix), cudaMemcpyHostToDevice));

    // Allocazione di memoria GPU per i blocchi
    ELLPACK_Block *d_blocks;
    checkCudaErrors(cudaMalloc(&d_blocks, hllMatrixHost->num_blocks * sizeof(ELLPACK_Block)));
    checkCudaErrors(cudaMemcpy(&d_hll_matrix->blocks, &d_blocks, sizeof(ELLPACK_Block *), cudaMemcpyHostToDevice));

      for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        ELLPACK_Block *block = &hllMatrixHost->blocks[i];

        int *d_JA=nullptr;
        double *d_AS=nullptr;

          // Supponiamo che 'size_of_arrays' sia il numero di elementi in 'JA' e 'AS'
        size_t JA_size = hllMatrixHost->blocks[i].size_of_arrays * sizeof(int);
        size_t AS_size = hllMatrixHost->blocks[i].size_of_arrays * sizeof(double);

        // Allocazione e copia di JA
        checkCudaErrors(cudaMalloc(&d_JA, JA_size));
        checkCudaErrors(cudaMemcpy(d_JA, block->JA, JA_size, cudaMemcpyHostToDevice));

        // Allocazione e copia di AS
        checkCudaErrors(cudaMalloc(&d_AS, AS_size));
        checkCudaErrors(cudaMemcpy(d_AS, block->AS, AS_size, cudaMemcpyHostToDevice));

        // Imposta i puntatori nel blocco GPU
        ELLPACK_Block d_block;
        d_block.JA = d_JA;
        d_block.AS = d_AS;
        d_block.max_nz_per_row = block->max_nz_per_row;
        d_block.nz_per_block = block->nz_per_block;

        checkCudaErrors(cudaMemcpy(&d_blocks[i], &d_block, sizeof(ELLPACK_Block), cudaMemcpyHostToDevice));
    }


    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);

    checkCudaErrors(cudaMalloc((void **) &d_x, matrix_data_host->M * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_x, x_h, matrix_data_host->M * sizeof(double), cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMalloc((void **) &d_y, matrix_data_host->M * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_y, y_h, matrix_data_host->M * sizeof(double), cudaMemcpyHostToDevice));


    // Ottieni il numero di Streaming Multiprocessors
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    int blocks_per_grid, threads_per_block;
    int num_rows = matrix_data_host->M; // Numero di righe della matrice

    configure_grid_warp(num_rows, sm_count, &blocks_per_grid, &threads_per_block);


    // Avvia il timer
    timer->start();

    // Invoca il kernel CUDA
    matvec_Hll_cuda<<<blocks_per_grid, threads_per_block>>>(d_hll_matrix, d_x, d_y, matrix_data_host->M);
    // Dopo il kernel CUDA, verifica errori
    checkCudaErrors(cudaDeviceSynchronize());

    // Ferma il timer
    timer->stop();

    matrixPerformance node{};
    node.seconds = timer->getTime()/1000.0f;
    node.flops = 0;
    node.gigaFlops = 0;

    //printf("HLLv1 time -> %lf\n", timer->getTime()/1000.0f);


    // Free delle risorse allocate
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        ELLPACK_Block *block = &hllMatrixHost->blocks[i];
        cudaFree(block->JA);
        cudaFree(block->AS);
    }
    cudaFree(d_blocks);
    cudaFree(d_hll_matrix);
    cudaFree(d_x);
    cudaFree(d_y);
    free(y_h);
    free(hllMatrixHost->blocks);
    free(hllMatrixHost);
    return node;
}

matrixPerformance parallel_hll_cuda_v2(matrixData *matrix_data_host, double *x_h) {
    double *d_y;
    double *d_x;
    int M = matrix_data_host->M;
    int maxThreadsPerBlock,maxGridDimX,grid_x, grid_y,numBlock;
    int max_nz_per_row_global = 0;
    // Pulizia della memoria GPU prima delle allocazioni
    cudaDeviceReset();
    // Controllo memoria iniziale
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

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
    convert_to_hll_cuda(matrix_data_host, hllMatrixHost);


        // Copia della struttura HLL_Matrix in memoria GPU
    HLL_Matrix *d_hll_matrix;
    checkCudaErrors(cudaMalloc(&d_hll_matrix, sizeof(HLL_Matrix)));
    checkCudaErrors(cudaMemcpy(d_hll_matrix, hllMatrixHost, sizeof(HLL_Matrix), cudaMemcpyHostToDevice));

    // Allocazione di memoria GPU per i blocchi
    ELLPACK_Block *d_blocks;
    checkCudaErrors(cudaMalloc(&d_blocks, hllMatrixHost->num_blocks * sizeof(ELLPACK_Block)));
    checkCudaErrors(cudaMemcpy(&d_hll_matrix->blocks, &d_blocks, sizeof(ELLPACK_Block *), cudaMemcpyHostToDevice));

    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        ELLPACK_Block *block = &hllMatrixHost->blocks[i];

        int *d_JA=nullptr;
        double *d_AS=nullptr;

          // Supponiamo che 'size_of_arrays' sia il numero di elementi in 'JA' e 'AS'
        size_t JA_size = hllMatrixHost->blocks[i].size_of_arrays * sizeof(int);
        size_t AS_size = hllMatrixHost->blocks[i].size_of_arrays * sizeof(double);

        // Allocazione e copia di JA
        checkCudaErrors(cudaMalloc(&d_JA, JA_size));
        checkCudaErrors(cudaMemcpy(d_JA, block->JA, JA_size, cudaMemcpyHostToDevice));

        // Allocazione e copia di AS
        checkCudaErrors(cudaMalloc(&d_AS, AS_size));
        checkCudaErrors(cudaMemcpy(d_AS, block->AS, AS_size, cudaMemcpyHostToDevice));

        // Imposta i puntatori nel blocco GPU
        ELLPACK_Block d_block;
        d_block.JA = d_JA;
        d_block.AS = d_AS;
        d_block.max_nz_per_row = block->max_nz_per_row;
        d_block.nz_per_block = block->nz_per_block;

        checkCudaErrors(cudaMemcpy(&d_blocks[i], &d_block, sizeof(ELLPACK_Block), cudaMemcpyHostToDevice));
    }


    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);

    checkCudaErrors(cudaMalloc((void **) &d_x, matrix_data_host->M * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_x, x_h, matrix_data_host->M * sizeof(double), cudaMemcpyHostToDevice));
    //printf("Copiato x in GPU\n");

    checkCudaErrors(cudaMalloc((void **) &d_y, matrix_data_host->M * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_y, y_h, matrix_data_host->M * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_y, 0, matrix_data_host->M * sizeof(double)));



    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    cudaDeviceGetAttribute(&maxGridDimX, cudaDevAttrMaxGridDimX, 0);


    numBlock = hllMatrixHost->num_blocks;


    // L'obiettivo è calcolare e allineare le dimensioni della griglia e
    // del blocco per ottenere un'efficiente distribuzione del carico di lavoro.
    // La configurazione della griglia viene ottimizzata per:
    // - adattarsi ai limiti hardware,
    // - garantire che i blocchi e il numero massimo di non zeri per riga siano multipli di HackSize,
    // - assicurarsi che la griglia non superi le dimensioni massime supportate dalla GPU,
    // - migliorare il bilanciamento del lavoro suddividendo la griglia in due dimensioni.


    // Trova il massimo numero di non zeri per riga nei blocchi
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        if (hllMatrixHost->blocks[i].max_nz_per_row > max_nz_per_row_global) {
            max_nz_per_row_global = hllMatrixHost->blocks[i].max_nz_per_row;
        }
    }

    if (max_nz_per_row_global % HackSize != 0) {
        max_nz_per_row_global = ((max_nz_per_row_global / HackSize) + 1) * HackSize;
    }
    if (max_nz_per_row_global > maxThreadsPerBlock / HackSize) {
        max_nz_per_row_global = maxThreadsPerBlock / HackSize;
    }

    // Allinea num_blocks a multiplo di HackSize senza superare maxGridDimX
    if (numBlock % HackSize != 0) {
        numBlock = ((numBlock / HackSize) + 1) * HackSize;
    }
    if (numBlock > maxGridDimX) {
        numBlock = maxGridDimX;
    }

    // Distribuzione della griglia su due dimensioni per migliorare la parallelizzazione
    grid_x = (int)sqrt((float)numBlock);
    grid_y = (numBlock + grid_x - 1) / grid_x; // Arrotonda all'intero superiore


    dim3 BLOCK_DIM1(HackSize, HackSize);
    dim3 GRID_DIM1(grid_x, grid_y);


    // Avvia il timer
    timer->start();
    // Invoca il kernel CUDA
    matvec_Hll_cuda_SH<<<GRID_DIM1, BLOCK_DIM1>>>(d_hll_matrix, d_x, d_y, M);
    // Dopo il kernel CUDA, verifica errori
    checkCudaErrors(cudaDeviceSynchronize());

    // Ferma il timer
    timer->stop();

    checkCudaErrors(cudaMemcpy(y_h, d_y,  M* sizeof(double), cudaMemcpyDeviceToHost));


    matrixPerformance node{};
    node.seconds = timer->getTime()/1000.0f;
    node.flops = 0;
    node.gigaFlops = 0;
    node.relativeError = checkDifferencesCUDA(y_h , matrix_data_host->M);

    //printf("HLLv2 time -> %lf\n", timer->getTime()/1000.0f);

    // Free delle risorse allocate
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        ELLPACK_Block *block = &hllMatrixHost->blocks[i];
        cudaFree(block->JA);
        cudaFree(block->AS);
    }
    cudaFree(d_blocks);
    cudaFree(d_hll_matrix);
    cudaFree(d_x);
    cudaFree(d_y);
    free(y_h);
    free(hllMatrixHost->blocks);
    free(hllMatrixHost);
    return node;
}
