#include <time.h>
#include <stdio.h>

#include "../libs/csrOperations.h"

#include <omp.h>
#include <stdlib.h>
#include "../libs/csrTool.h"
#include "../libs/data_structure.h"

/* Funzione per svolgere il prodotto matrice-vettore, con memorizzazione CSR della matrice, in modo serializzato */
struct matrixPerformance serial_csr(struct matrixData *matrix_data, double *x) {
    int *IRP, *JA;
    double *AS;

    /* Vettore di output del risultato y <- Ax */
    double *y = malloc((size_t)matrix_data->M * sizeof(double));
    if (y == NULL) {
        printf("Errore nell'allocazione della memoria per il vettore di output y\n");
        exit(EXIT_FAILURE);
    }

    /* Conversione in formato CSR */
    convert_to_csr(matrix_data->M, matrix_data->nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, &IRP, &JA, &AS);
    for (int i = 0; i < matrix_data->M; i++) {
        y[i] = 0.0;
    }
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matvec_csr(matrix_data->M, IRP, JA, AS, x, y);
    clock_gettime(CLOCK_MONOTONIC, &end);

    const double time_spent = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

    struct matrixPerformance node;
    node.seconds = time_spent;
    node.flops = 0;
    node.megaFlops = 0;

    free(y);
    free(IRP);
    free(JA);
    free(AS);

    return node;
}

/* Funzione per calcolare una soglia dinamica per nz_per_thread_threshold */
int calculate_nz_per_thread_threshold(int M, int N, int nz) {
    if (M <= 0 || N <= 0 || nz <= 0) {
        printf("Errore: Parametri non validi per calculate_nz_per_thread_threshold (M: %d, N: %d, nz: %d)\n", M, N, nz);
        exit(EXIT_FAILURE);
    }

    double base_threshold = 500.0;
    double dynamic_threshold = base_threshold * (1 + ((double)nz / (M * N)));
    return (int)dynamic_threshold;
}

/* Funzione per calcolare il numero ideale di thread */
int calculate_optimal_threads(int M, int N, int nz) {
    if (M <= 0 || N <= 0 || nz <= 0) {
        printf("Errore: Parametri non validi per calculate_optimal_threads (M: %d, N: %d, nz: %d)\n", M, N, nz);
        exit(EXIT_FAILURE);
    }

    int max_threads = omp_get_max_threads();

    // Calcolo della soglia dinamica per nz_per_thread_threshold
    int nz_per_thread_threshold = calculate_nz_per_thread_threshold(M, N, nz);

    // Calcolo dinamico della soglia minima di righe per thread
    int rows_per_thread_threshold = (M > 100) ? (M / 10) : 1; // Minimo 10% di righe per thread o almeno 1

    // Calcolo del numero ideale di thread
    int threads_based_on_nz = nz / nz_per_thread_threshold;
    int threads_based_on_rows = M / rows_per_thread_threshold;

    int optimal_threads = (threads_based_on_nz < threads_based_on_rows) ? threads_based_on_nz : threads_based_on_rows;

    // Assicura almeno 3 thread se il numero medio di non zeri per riga è disparo, 4 se è pari
    int avg_nnz_per_row = nz / M;
    if (optimal_threads < 2) {
        optimal_threads = (avg_nnz_per_row % 2 == 0) ? 4 : 3;
    }
    if (optimal_threads > max_threads) optimal_threads = max_threads;

    printf("Numero di thread calcolato: %d (max_threads: %d, avg_nnz_per_row: %d)\n", optimal_threads, max_threads, avg_nnz_per_row);

    return optimal_threads;
}

void compute_thread_row_partition(int M, int nz, int *num_threads, int *IRP, int **start_row, int **end_row) {
    *start_row = malloc((size_t)(*num_threads) * sizeof(int));
    *end_row = malloc((size_t)(*num_threads) * sizeof(int));
    int *nnz_per_thread_count = malloc((size_t)(*num_threads) * sizeof(int)); // Array per conteggio dei non zeri

    if (*start_row == NULL || *end_row == NULL || nnz_per_thread_count == NULL) {
        printf("Errore nell'allocazione della memoria per la partizione delle righe\n");
        exit(EXIT_FAILURE);
    }

    // Inizializza i valori
    for (int t = 0; t < *num_threads; t++) {
        (*start_row)[t] = -1;
        (*end_row)[t] = -1;
        nnz_per_thread_count[t] = 0; // Inizializza il conteggio a 0
    }

    int ideal_nnz_per_thread = nz / *num_threads; // Quota ideale di non zeri per ogni thread
    int current_thread = 0;
    int current_nnz = 0;

    // Assegna righe in modo da bilanciare i non zeri
    for (int i = 0; i < M; i++) {
        int nnz_in_row = IRP[i + 1] - IRP[i]; // Non zeri nella riga corrente

        if ((*start_row)[current_thread] == -1) {
            (*start_row)[current_thread] = i; // Imposta l'inizio per il thread corrente
        }

        current_nnz += nnz_in_row;
        nnz_per_thread_count[current_thread] += nnz_in_row;

        // Se il limite di non zeri per il thread è raggiunto, passa al prossimo thread
        if (current_nnz >= ideal_nnz_per_thread && current_thread < *num_threads - 1) {
            (*end_row)[current_thread] = i + 1; // Fine inclusiva per il thread corrente
            current_thread++;
            current_nnz = 0;
        }
    }

    // Assegna la fine per l'ultimo thread
    if (current_thread < *num_threads) {
        (*end_row)[current_thread] = M;
    }

    // Rimuovi i thread non validi (ad esempio, se ci sono più thread rispetto alle righe o i non zeri)
    int valid_threads = 0;
    for (int t = 0; t < *num_threads; t++) {
        if ((*start_row)[t] != -1 && (*end_row)[t] != -1 && nnz_per_thread_count[t] > 0) {
            // Mantieni il thread valido
            (*start_row)[valid_threads] = (*start_row)[t];
            (*end_row)[valid_threads] = (*end_row)[t];
            nnz_per_thread_count[valid_threads] = nnz_per_thread_count[t];
            valid_threads++;
        }
    }

    // Aggiorna il numero finale di thread
    *num_threads = valid_threads;

    // Stampa il risultato per ogni thread
    int result=0;
    for (int t = 0; t < *num_threads; t++) {
        printf("Thread %d: righe [%d, %d), non zeri = %d , numero di righe: %d\n",
               t, (*start_row)[t], (*end_row)[t], nnz_per_thread_count[t], M);
        result=result+nnz_per_thread_count[t];
    }
    printf("non zeri nella matrice: %d , numero di non zeri assegnati:%d\n",nz,result);
    free(nnz_per_thread_count); // Libera la memoria allocata
}

struct matrixPerformance parallel_csr(struct matrixData *matrix_data, double *x) {
    int *IRP, *JA;
    double *AS;

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

    convert_to_csr(matrix_data->M, matrix_data->nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, &IRP, &JA, &AS);

    int num_threads= calculate_optimal_threads(matrix_data->M,matrix_data->N, matrix_data->nz);
    printf("numero di thread: %d\n",num_threads);
    int *start_row, *end_row;


    compute_thread_row_partition(matrix_data->M, matrix_data->nz, &num_threads, IRP, &start_row, &end_row);

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    matvec_csr_openMP(IRP, JA, AS, x, y, start_row, end_row, num_threads, matrix_data->nz, matrix_data->M);
    clock_gettime(CLOCK_MONOTONIC, &end);

    const double time_spent = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    node.seconds = time_spent;

    free(start_row);
    free(end_row);
    free(y);
    free(IRP);
    free(JA);
    free(AS);

    return node;
}