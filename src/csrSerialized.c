#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "../libs/csrSerialized.h"
#include "../libs/csrTool.h"
#include "../libs/data_structure.h"

// Funzione serial_csr
struct matrixPerformance serial_csr(int M, int N, int nz, int *row_indices, int *col_indices, double *values, double *x) {
    int *IRP, *JA;
    double *AS;

    /* Vettore di output del risultato y <- Ax */
    double *y = malloc((size_t)M * sizeof(double));
    if (y == NULL) {
        printf("Errore: Memoria non disponibile per il vettore y!\n");
        exit(EXIT_FAILURE); // Esce con un errore
    }

    // Conversione in formato CSR
    convert_to_csr(M, N, nz, row_indices, col_indices, values, &IRP, &JA, &AS);

    struct timespec start, end;

    // Misurazione del tempo
    clock_gettime(CLOCK_MONOTONIC, &start);
    matvec_csr(M, IRP, JA, AS, x, y);
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calcolo del tempo e delle performance
    const double time_spent = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;


    // Stampa del risultato
    printf("Risultato del prodotto matrice-vettore:\n");
    for (int j = 0; j < M; j++) {
        printf("y[%d] = %.2f\n", j, y[j]);
    }

    printf("Tempo per il prodotto matrice-vettore: %.6f secondi\n", time_spent);


    // Creazione della struttura con i risultati
    struct matrixPerformance node;
    node.seconds = time_spent;

    // Liberazione della memoria
    free(y);
    free(IRP);
    free(JA);
    free(AS);

    return node; // Ritorna la struttura
}
