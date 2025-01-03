#include <time.h>
#include <stdio.h>

#include "../libs/csrOperations.h"

#include <omp.h>
#include <stdlib.h>
#include <tgmath.h>

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

/* Funzione per calcolare il numero di non-zeri per riga */
void calculate_nonzeros_per_row(const int M, const int *IRP, int *nonzeros_per_row) {
    for (int i = 0; i < M; i++) {
        nonzeros_per_row[i] = IRP[i + 1] - IRP[i];
    }
}

/**
 * Funzione: distribute_work_balanced
 * ----------------------------------
 * Questa funzione suddivide le righe di una matrice sparse CSR (Compressed Sparse Row)
 * tra un numero specificato di thread, bilanciando il carico in base al numero di elementi
 * non nulli (non-zeri) in ciascuna riga. L'obiettivo è assegnare a ciascun thread un numero
 * complessivo di non-zeri il più simile possibile al valore medio, minimizzando il carico
 * computazionale sbilanciato tra i thread.

 * La funzione assegna progressivamente le righe a ciascun thread fino a raggiungere o
 * superare il carico medio di non-zeri per thread. Gli ultimi thread gestiscono eventuali
 * righe residue, garantendo che tutte le righe vengano elaborate.

 * Alla fine, la funzione restituisce un array bidimensionale (`thread_rows`), in cui
 * `thread_rows[t]` rappresenta l'elenco delle righe assegnate al thread `t`.
 * Inoltre, la funzione aggiorna l'array `row_counts` per indicare quante righe
 * sono state assegnate a ciascun thread.

 * Input:
 *  - M: Numero totale di righe nella matrice.
 *  - nonzeros_per_row: Array che contiene il numero di non-zeri per ciascuna riga.
 *  - row_counts: Array di output per salvare il numero di righe assegnate a ciascun thread.
 *  - num_threads: Numero di thread da utilizzare per il calcolo parallelo.

 * Output:
 *  - thread_rows: Un array bidimensionale dove thread_rows[t] contiene l'elenco
 *    delle righe assegnate al thread `t`.

 * Debugging:
 *  - La funzione stampa per ciascun thread le righe assegnate e il numero totale di
 *    non-zeri corrispondenti.
 */


int** distribute_work_balanced(const int M, const int *nonzeros_per_row, int *row_counts, const int num_threads) {
    int total_nonzeros = 0;

    // Calcolo del numero totale di non-zeri
    for (int i = 0; i < M; i++) {
        total_nonzeros += nonzeros_per_row[i];
    }

    // Calcolo del carico medio di non-zeri per thread
    int avg_nonzeros_per_thread = total_nonzeros / num_threads;
    int remainder_nonzeros = total_nonzeros % num_threads; // Gestione del resto

    // Array di array per le righe di ogni thread
    int** thread_rows = malloc(num_threads * sizeof(int*));
    if (!thread_rows) {
        perror("Errore nell'allocazione di thread_rows");
        exit(1);
    }

    for (int t = 0; t < num_threads; t++) {
        thread_rows[t] = malloc(M * sizeof(int)); // Ogni thread può gestire al massimo tutte le righe
        if (!thread_rows[t]) {
            perror("Errore nell'allocazione delle righe del thread");
            exit(1);
        }
    }

    int current_thread = 0;
    int accumulated_nonzeros = 0;

    // Inizializza il contatore delle righe per ogni thread
    for (int t = 0; t < num_threads; t++) {
        row_counts[t] = 0;
    }

    // Distribuzione delle righe basata sul numero di non-zeri
    for (int i = 0; i < M; i++) {
        // Aggiungi la riga corrente al thread
        thread_rows[current_thread][row_counts[current_thread]++] = i;
        accumulated_nonzeros += nonzeros_per_row[i];

        // Passa al thread successivo se il carico medio è raggiunto o superato
        if ((accumulated_nonzeros >= avg_nonzeros_per_thread + (current_thread < remainder_nonzeros ? 1 : 0)) &&
            current_thread < num_threads - 1) {
            current_thread++;
            accumulated_nonzeros = 0; // Resetta il contatore per il thread successivo
            }
    }
    // Stampa la distribuzione delle righe (debugging)
    for (int t = 0; t < num_threads; t++) {
        printf("Thread %d: Righe assegnate: ", t);
        int thread_nonzeros = 0;
        for (int j = 0; j < row_counts[t]; j++) {
            printf("%d ", thread_rows[t][j]);
            if (thread_rows[t][j]!=0) {
                thread_nonzeros ++ ;
            }

        }
        printf("Numero di non-zeri assegnati: %d\n", thread_nonzeros);
    }

    return thread_rows; // Restituisce l'array bidimensionale
}


/**
 * Funzione: parallel_csr
 * ----------------------
 * Questa funzione esegue il prodotto matrice-vettore parallelo per una matrice sparse in formato CSR.
 * La funzione utilizza OpenMP per sfruttare il calcolo parallelo, determinando dinamicamente il numero
 * ottimale di thread da utilizzare per garantire un buon bilanciamento del carico e prestazioni efficienti.
 *
 * La determinazione del numero ideale di thread avviene in base alle seguenti considerazioni:
 *
 * 1. **Numero di non-zeri per riga (nz_per_row)**:
 *    - Viene calcolato il numero medio di non-zeri per riga come `nz_per_row = nz / M`, dove `nz` è il
 *      numero totale di non-zeri e `M` è il numero totale di righe.
 *
 * 2. **Numero minimo di righe per thread (min_rows_per_thread)**:
 *    - Per garantire che ogni thread abbia un carico di lavoro significativo, si richiede che ogni thread
 *      elabori almeno 30 non-zeri. Il numero minimo di righe per thread è calcolato come:
 *        `min_rows_per_thread = ceil(30 / nz_per_row)`
 *
 * 3. **Numero massimo di thread dinamico (max_threads_dynamic)**:
 *    - In base al numero minimo di righe per thread, viene calcolato il massimo numero di thread che possono
 *      essere utilizzati senza sovraccaricare i thread con un numero insufficiente di righe:
 *        `max_threads_dynamic = M / min_rows_per_thread`
 *    - Per gestire i casi in cui `M % min_rows_per_thread != 0` (righe residue non divisibili equamente),
 *      si aggiunge un thread extra per gestire tali righe residue:
 *        `if (M % min_rows_per_thread != 0) max_threads_dynamic += 1`
 *
 * 4. **Limite massimo e minimo del numero di thread**:
 *    - Il numero di thread viene limitato al massimo tra `omp_get_max_threads()` e il numero di righe `M`.
 *    - Se il numero calcolato è inferiore a 1, viene forzato l'uso di almeno 1 thread.
 *
 * 5. **Scelta finale del numero di thread (num_threads)**:
 *    - Il numero finale di thread è scelto come il minimo tra `max_threads_dynamic` e il numero massimo di
 *      thread disponibili (`omp_get_max_threads()`), garantendo che ogni thread riceva un carico significativo.
 *
 * Input:
 *  - matrix_data: Una struttura contenente le informazioni della matrice sparse (righe, colonne, valori, ecc.).
 *  - x: Il vettore di input per il prodotto matrice-vettore.
 *
 * Output:
 *  - struct matrixPerformance: Una struttura che contiene informazioni sulle prestazioni, inclusi i tempi
 *    di esecuzione.
 *
 * Debugging:
 *  - La funzione stampa il numero massimo di thread disponibili, il numero ottimale di thread calcolato
 *    (dopo eventuali aggiustamenti modulari), il numero minimo di righe per thread, e il numero medio di
 *    non-zeri per riga.
 */

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
    convert_to_csr(matrix_data->M, matrix_data->nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, &IRP, &JA, &AS);

    // Calcolo dei non-zeri per riga
    int *nonzeros_per_row = malloc(matrix_data->M * sizeof(int));
    if (!nonzeros_per_row) {
        perror("Errore nell'allocazione della memoria per nonzeros_per_row.");
        exit(1);
    }
    calculate_nonzeros_per_row(matrix_data->M, IRP, nonzeros_per_row);

    // Determinazione dinamica del numero di thread
    int max_threads = omp_get_max_threads();
    int M = matrix_data->M;    // Numero totale di righe
    int nz = matrix_data->nz;  // Numero totale di non-zeri

    // Calcolo del numero medio di non-zeri per riga
    double nz_per_row = (double)nz / M;

    // Calcolo del numero minimo di righe per thread per garantire almeno X non-zeri per thread
    int min_rows_per_thread = (int)ceil(30.0 / nz_per_row);
    printf("Calcolo del numero minimo di righe per thread per garantire almeno X non-zeri per thread: %d\n", min_rows_per_thread);

    // Calcolo del numero massimo di thread dinamico
    int max_threads_dynamic = M / min_rows_per_thread;

    // Limita il numero massimo di thread a quelli disponibili
    if (max_threads_dynamic > max_threads) {
        max_threads_dynamic = max_threads; // Non possiamo superare i thread disponibili
    }

    // Garantisci che almeno un thread venga utilizzato
    if (max_threads_dynamic < 1) {
        max_threads_dynamic = 1;
    }

    int num_threads = max_threads_dynamic;

    if (!mm_is_pattern(matrix_data->matcode) &&
        !mm_is_symmetric(matrix_data->matcode) &&
        mm_is_coordinate(matrix_data->matcode)) {
        printf("Numero massimo di thread disponibili: %d\n", max_threads);
        printf("Numero ottimale di thread calcolato: %d\n", num_threads);
        printf("Numero medio di non-zeri per riga se uniformemente distribuiti: %.2f\n", nz_per_row);

        // Suddivisione del lavoro tra i thread
        int *row_counts = malloc(num_threads * sizeof(int)); // Numero di righe per ogni thread
        int **thread_rows = distribute_work_balanced(M, nonzeros_per_row, row_counts, num_threads);

        // Calcolo parallelo
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        matvec_csr_openMP(IRP, JA, AS, x, y, thread_rows, row_counts, num_threads, M);
        clock_gettime(CLOCK_MONOTONIC, &end);

        const double time_spent = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
        node.seconds = time_spent;

        // Libera la memoria
        for (int t = 0; t < num_threads; t++) {
            free(thread_rows[t]);
        }
        free(thread_rows);

        free(row_counts);

        }

    free(nonzeros_per_row);
    free(y);
    free(IRP);
    free(JA);
    free(AS);

    return node;
}