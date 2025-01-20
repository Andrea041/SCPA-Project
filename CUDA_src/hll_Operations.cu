#include "../libs/costants.h"
#include "../CUDA_libs/csrOperations.h"
#include "../CUDA_libs/cudaCostants.h"
#include "../CUDA_libs/hllTool.h"
#include "../libs/data_structure.h"

#include <helper_cuda.h>
#include <helper_timer.h>

void configure_grid_warp(int num_rows, int sm_count, int *blocks, int *threads) {
    // Definizione della dimensione del warp


    // Numero massimo di thread per blocco dalla GPU
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);

    // Numero massimo di thread per SM
    int max_threads_per_sm;
    cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

    // Configurazione dinamica dei thread per blocco
    int threads_per_block = (num_rows < max_threads_per_block) ? num_rows : max_threads_per_block;
    threads_per_block = (threads_per_block / WARP_SIZE) * WARP_SIZE; // Arrotonda al multiplo di warp_size

    if (threads_per_block == 0) {
        threads_per_block = WARP_SIZE; // Almeno un warp
    }

    // Numero di blocchi necessario
    int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;

    // Arrotonda il numero di blocchi al multiplo del numero di SM
    blocks_per_grid = ((blocks_per_grid + sm_count - 1) / sm_count) * sm_count;

    // Se il numero di blocchi è troppo grande, riducilo per evitare overhead
    if (blocks_per_grid * threads_per_block > max_threads_per_sm * sm_count) {
        blocks_per_grid = sm_count * max_threads_per_sm / threads_per_block;
    }

    // Configura i valori di output
    *blocks = blocks_per_grid;
    *threads = threads_per_block;

    printf("Configurazione griglia warp-aware: blocks_per_grid = %d, threads_per_block = %d\n", blocks_per_grid, threads_per_block);
}


// Configura la griglia dei blocchi e dei thread
/*void configure_grid_warp(int M, int sm_count, int *blocks, int *threads) {
  //  printf("Configurazione griglia: M=%d, sm_count=%d\n", M, sm_count);
    int total_threads = ((M + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE; // Allinea a warp
    *threads = WARP_SIZE;  // Ogni blocco ha un warp
    *blocks = (total_threads + *threads - 1) / *threads;
  //  printf("Threads per blocco: %d, Numero di blocchi: %d\n", *threads, *blocks);

    // Assicura che il numero di warp sia multiplo di SM count
    if (*blocks % sm_count != 0) {
        *blocks = ((*blocks / sm_count) + 1) * sm_count;
    }

    printf("Configurazione griglia warp-aware: blocks_per_grid = %d, threads_per_block = %d\n", *blocks,    *threads);
}
*/
// Funzione principale per calcolare il prodotto parallelo
matrixPerformance parallel_hll_cuda(matrixData *matrix_data_host, double *x_h) {
    double *d_y;
    double *d_x;
    int M = matrix_data_host->M;

    // Pulizia della memoria GPU prima delle allocazioni
    cudaDeviceReset();
    // Controllo memoria iniziale
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Memoria GPU disponibile dopo reset: %lu / %lu bytes\n", free_mem, total_mem);

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
    convert_to_hll_cuda(matrix_data_host, hllMatrixHost, hllMatrixHost);


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
    //printf("Copiato y iniziale in GPU\n");

    // Ottieni il numero di Streaming Multiprocessors
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
   // printf(" sm_count: %d\n", sm_count);


    //debug usando una versione seriale del calcolo in hll, la memorizzazione funzina correttamente.


    /*int threads_per_block = 256; // Numero standard di thread per blocco
    int blocks_per_grid = (M + threads_per_block - 1) / threads_per_block;

    printf("Configurazione automatica: blocks_per_grid = %d, threads_per_block = %d\n", blocks_per_grid, threads_per_block);

    const dim3 GRID_DIM(blocks_per_grid);*/
    // Configura la griglia automaticamente
    int blocks_per_grid, threads_per_block;
    int num_rows = matrix_data_host->M; // Numero di righe della matrice

    configure_grid_warp(num_rows, sm_count, &blocks_per_grid, &threads_per_block);

    const dim3 GRID_DIM(blocks_per_grid);

    // Avvia il timer
    timer->start();

    // Invoca il kernel CUDA
    matvec_Hll_cuda<<<GRID_DIM, threads_per_block>>>(d_hll_matrix, d_x, d_y, matrix_data_host->M);
    // Dopo il kernel CUDA, verifica errori
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Sincronizzazione e controllo degli errori
    checkCudaErrors(cudaDeviceSynchronize());

    // Ferma il timer
    timer->stop();
/*
    checkCudaErrors(cudaMemcpy(y_h, d_y,  matrix_data_host->M * sizeof(double), cudaMemcpyDeviceToHost));
    printf("Risultato copiato da GPU a CPU:\n");
    for (int i = 0; i < matrix_data_host->M; i++) {
        printf("y[%d] = %lf\n", i, y_h[i]);
    }
*/
    matrixPerformance node{};
    node.seconds = timer->getTime();
    node.flops = 0;
    node.gigaFlops = 0;


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
