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

/* Funzione per suddividere il lavoro in base a un numero medio di non-zeri per riga equilibrato */
void distribute_work_balanced(const int M, const int *nonzeros_per_row, int *row_start, int *row_end, const int num_threads) {
    int total_nonzeros = 0;

    // Calcolo del numero totale di non-zeri
    for (int i = 0; i < M; i++) {
        total_nonzeros += nonzeros_per_row[i];
    }

    // Calcolo del carico medio di non-zeri per thread
    int avg_nonzeros_per_thread = total_nonzeros / num_threads;

    int current_thread = 0; // Thread corrente
    int accumulated_nonzeros = 0; // Contatore per i non-zeri assegnati al thread corrente

    row_start[current_thread] = 0; // La prima riga inizia sempre da 0

    for (int i = 0; i < M; i++) {
        accumulated_nonzeros += nonzeros_per_row[i]; // Aggiungi i non-zeri della riga corrente

        // Se il carico del thread supera o è vicino al carico medio e non siamo all'ultimo thread
        if (accumulated_nonzeros >= avg_nonzeros_per_thread && current_thread < num_threads - 1) {
            row_end[current_thread] = i; // Assegna l'ultima riga al thread corrente
            current_thread++; // Passa al thread successivo
            row_start[current_thread] = i + 1; // La riga successiva inizia dal thread successivo
            accumulated_nonzeros = 0; // Resetta il contatore per il thread successivo
        }
    }

    // L'ultimo thread gestisce le righe rimanenti
    row_end[current_thread] = M - 1;
}



/* Implementazione della funzione parallel_csr */
struct matrixPerformance parallel_csr(struct matrixData *matrix_data, double *x) {
    int *IRP, *JA;
    double *AS;

    /* Vettore di output del risultato y <- Ax */
    double *y = malloc((size_t)matrix_data->M * sizeof(double));
    if (y == NULL) {
        printf("Errore nell'allocazione della memoria per il vettore di output y\n");
        exit(EXIT_FAILURE);
    }

    struct matrixPerformance node;
    /* Conversione in formato CSR */
    convert_to_csr(matrix_data->M, matrix_data->nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, &IRP, &JA, &AS);

    node.seconds = 0;
    node.flops = 0;
    node.megaFlops = 0;
    if (mm_is_coordinate(matrix_data->matcode) && !mm_is_symmetric(matrix_data->matcode) && !mm_is_pattern(matrix_data->matcode)) {

        /* Calcolo dei non-zeri per riga */
        int *nonzeros_per_row = malloc(matrix_data->M * sizeof(int));
        if (!nonzeros_per_row ) {
            perror("Errore nell'allocazione della memoria per nonzeros_per_row.");
            exit(1);
        }
        calculate_nonzeros_per_row(matrix_data->M, IRP, nonzeros_per_row);

        printf("numero non zeri : %d\n",*nonzeros_per_row);


        /* Suddivisione del lavoro tra i thread */
        int num_threads = omp_get_max_threads();
        int *row_start = malloc(num_threads * sizeof(int));
        int *row_end = malloc(num_threads * sizeof(int));
        distribute_work_balanced(matrix_data->M, nonzeros_per_row, row_start, row_end, num_threads);

        /* Esecuzione parallela del prodotto matrice-vettore */
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        matvec_csr_openMP ( IRP, JA, AS, x, y, row_start, row_end, num_threads) ;
        clock_gettime(CLOCK_MONOTONIC, &end);

        const double time_spent = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

        node.seconds = time_spent;

        /* Pulizia della memoria */
        free(y);
        free(IRP);
        free(JA);
        free(AS);
        free(nonzeros_per_row);
        free(row_start);
        free(row_end);

    } /*else if (mm_is_symmetric(matrix_data->matcode)) {

    } else if () {

    }*/

    return node;
}
