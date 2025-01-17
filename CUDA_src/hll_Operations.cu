#include "../CUDA_libs/csrTool.h"
#include "../CUDA_libs/csrOperations.h"
#include "../CUDA_libs/cudaCostants.h"
#include "../CUDA_libs/hllTool.h"
#include "../libs/data_structure.h"

#include <helper_cuda.h>
#include <helper_timer.h>

// Funzione sequenziale per il prodotto matrice-vettore usando la memorizzazione HLL in ELLPACK
void sequentialMatVec_Hll(HLL_Matrix *hll_matrix, double *x, double *y, int M) {
    for (int block_id = 0; block_id < hll_matrix->num_blocks; block_id++) {
        /* Calcolo delle righe di inizio e fine del blocco */
        int start_row = block_id * HackSize;
        int end_row = (block_id + 1) * HackSize;
        if (end_row > M) end_row = M;

        int row_offset = 0;
        /* Scorrimento delle righe di un unico blocco */
        for (int i = start_row; i < end_row; i++) {
            y[i] = 0.0;
            /* Scorrimento della riga selezionata (sarà lunga maxnz) */
            for (int j = 0; j < hll_matrix->blocks[block_id].max_nz_per_row; j++) {
                y[i] += hll_matrix->blocks[block_id].AS[j + row_offset] * x[hll_matrix->blocks[block_id].JA[j + row_offset]];
            }
            /* Incremento dell'offset per passare alla riga successiva */
            row_offset += hll_matrix->blocks[block_id].max_nz_per_row;
        }
    }
}


void configure_grid_warp(int num_rows, int sm_count, int *blocks, int *threads) {
    // Definizione della dimensione del warp
    const int warp_size = 32;

    // Numero massimo di thread per blocco dalla GPU
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);

    // Numero massimo di thread per SM
    int max_threads_per_sm;
    cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

    // Configurazione dinamica dei thread per blocco
    int threads_per_block = (num_rows < max_threads_per_block) ? num_rows : max_threads_per_block;
    threads_per_block = (threads_per_block / warp_size) * warp_size; // Arrotondare al multiplo di warp_size

    if (threads_per_block == 0) {
        threads_per_block = warp_size; // Almeno un warp se num_rows è molto piccolo
    }

    // Numero di blocchi necessari
    int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;

    // Arrotondare il numero di blocchi al multiplo del numero di SM
    blocks_per_grid = ((blocks_per_grid + sm_count - 1) / sm_count) * sm_count;

    // Configura i valori di output
    *blocks = blocks_per_grid;
    *threads = threads_per_block;

    printf("Configurazione griglia warp-aware: blocks_per_grid = %d, threads_per_block = %d\n", blocks_per_grid, threads_per_block);
}

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
        for (int j = 0; j < matrix_data_host->M * block->max_nz_per_row; j++) {
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
    checkCudaErrors(cudaMalloc(&d_hll_matrix, sizeof(HLL_Matrix)));
    checkCudaErrors(cudaMemcpy(d_hll_matrix, hllMatrixHost, sizeof(HLL_Matrix), cudaMemcpyHostToDevice));

    // Allocazione di memoria GPU per i blocchi
    ELLPACK_Block *d_blocks;
    checkCudaErrors(cudaMalloc(&d_blocks, hllMatrixHost->num_blocks * sizeof(ELLPACK_Block)));
    checkCudaErrors(cudaMemcpy(&d_hll_matrix->blocks, &d_blocks, sizeof(ELLPACK_Block *), cudaMemcpyHostToDevice));

    // Iterazione per copiare i blocchi uno per uno
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        ELLPACK_Block *block = &hllMatrixHost->blocks[i];

        int *d_JA;
        double *d_AS;

        // Allocazione e copia di JA
        checkCudaErrors(cudaMalloc(&d_JA, block->max_nz_per_row * matrix_data_host->M * sizeof(int)));
        checkCudaErrors(cudaMemcpy(d_JA, block->JA, block->max_nz_per_row * matrix_data_host->M * sizeof(int), cudaMemcpyHostToDevice));

        // Allocazione e copia di AS
        checkCudaErrors(cudaMalloc(&d_AS, block->max_nz_per_row * matrix_data_host->M * sizeof(double)));
        checkCudaErrors(cudaMemcpy(d_AS, block->AS, block->max_nz_per_row * matrix_data_host->M * sizeof(double), cudaMemcpyHostToDevice));

        // Imposta i puntatori nel blocco GPU
        ELLPACK_Block d_block;
        d_block.JA = d_JA;
        d_block.AS = d_AS;
        d_block.max_nz_per_row = block->max_nz_per_row;
        d_block.nz_per_block = block->nz_per_block;

        checkCudaErrors(cudaMemcpy(&d_blocks[i], &d_block, sizeof(ELLPACK_Block), cudaMemcpyHostToDevice));
        printf("Blocco %d copiato in GPU\n", i);

        // Debug: Verifica i dati copiati su GPU
        auto debug_JA = static_cast<int *>(malloc(block->max_nz_per_row * matrix_data_host->M * sizeof(int)));
        auto *debug_AS = static_cast<double *>(malloc(block->max_nz_per_row * matrix_data_host->M * sizeof(double)));

        checkCudaErrors(cudaMemcpy(debug_JA, d_JA, block->max_nz_per_row * matrix_data_host->M * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(debug_AS, d_AS, block->max_nz_per_row * matrix_data_host->M * sizeof(double), cudaMemcpyDeviceToHost));

        printf("JA (GPU -> CPU): ");
        for (int j = 0; j < block->max_nz_per_row * matrix_data_host->M; j++) {
            printf("%d ", debug_JA[j]);
        }
        printf("\nAS (GPU -> CPU): ");
        for (int j = 0; j < block->max_nz_per_row * matrix_data_host->M; j++) {
            printf("%lf ", debug_AS[j]);
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

    // Ottieni il numero di Streaming Multiprocessors
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA sm_count %d\n", sm_count);



    /*double *x = (double *)malloc(matrix_data_host->N * sizeof(double));
    double *y = (double *)malloc(M * sizeof(double));

    // Inizializza il vettore x
    for (int i = 0; i < matrix_data_host->N ; i++) {
        x[i] = 1.0; // Esempio di inizializzazione
    }

    // Calcolo sequenziale
    sequentialMatVec_Hll(hllMatrixHost, x, y, M);

    // Stampa del risultato
    printf("Risultato del prodotto matrice-vettore:\n");
    for (int i = 0; i < M; i++) {
        printf("y[%d] = %f\n", i, y[i]);
    }*/
    //debug usando una versione seriale del calcolo in hll, la memorizzazione funzina correttamente.

    // Configura la griglia


   /* configure_grid_warp(num_rows, sm_count, &blocks_per_grid, &threads_per_block);

    const dim3 GRID_DIM(blocks_per_grid);

   */
    int threads_per_block = 256; // Numero standard di thread per blocco
    int blocks_per_grid = (M + threads_per_block - 1) / threads_per_block;

    printf("Configurazione automatica: blocks_per_grid = %d, threads_per_block = %d\n", blocks_per_grid, threads_per_block);

    const dim3 GRID_DIM(blocks_per_grid);




    // Avvia il timer
    timer->start();

    // Invoca il kernel CUDA
    matvec_Hll_cuda<<<GRID_DIM, threads_per_block>>>(d_hll_matrix, d_x, d_y, matrix_data_host->M);

    // Sincronizzazione e controllo degli errori
    checkCudaErrors(cudaDeviceSynchronize());

    // Ferma il timer
    timer->stop();

   checkCudaErrors(cudaMemcpy(y_h, d_y,  matrix_data_host->M * sizeof(double), cudaMemcpyDeviceToHost));
    printf("Risultato copiato da GPU a CPU:\n");
    for (int i = 0; i < matrix_data_host->M; i++) {
        printf("y[%d] = %lf\n", i, y_h[i]);
    }

    matrixPerformance node{};
    node.seconds =0;// timer->getTime();
    node.flops = 0;
    node.gigaFlops = 0;


    //free(hllMatrixHost);

    checkCudaErrors(cudaFree(hllMatrixHost));

    return node;
}
