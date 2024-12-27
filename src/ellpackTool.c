#include <stdlib.h>
#include <stdio.h>

#include "../libs/ellpackTool.h"

/* Funzione per convertire la matrice in formato ELLPACK */
void convert_to_ellpack(int M, int nz, const int *row_indices, const int *col_indices, const double *values, int **JA, double **AS, int *MAXNZ) {
    *MAXNZ = 0;

    // Trova il massimo numero di non-zeri per riga
    int *row_counts = calloc(M, sizeof(int));
    if (row_counts == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria per row_counts.\n");
        exit(1);
    }

    for (int i = 0; i < nz; i++) {
        row_counts[row_indices[i]]++;
    }

    for (int i = 0; i < M; i++) {
        if (row_counts[i] > *MAXNZ) {
            *MAXNZ = row_counts[i];
        }
    }

    *JA = (int *)malloc(M * *MAXNZ * sizeof(int));
    *AS = (double *)malloc(M * *MAXNZ * sizeof(double));

    // Verifica l'allocazione della memoria
    if (*JA == NULL || *AS == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria per JA o AS.\n");
        exit(1);
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < *MAXNZ; j++) {
            (*JA)[i * *MAXNZ + j] = -1;  // Indice di colonna per celle vuote
            (*AS)[i * *MAXNZ + j] = 0.0;  // Valore zero per celle vuote
        }
    }

    // Popola JA e AS
    int *row_position = calloc(M, sizeof(int));
    if (row_position == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria per row_position.\n");
        exit(1);
    }

    for (int i = 0; i < nz; i++) {
        const int row = row_indices[i];
        const int pos = row_position[row];
        (*JA)[row * *MAXNZ + pos] = col_indices[i] + 1;  // Indici 1-based
        (*AS)[row * *MAXNZ + pos] = values[i];
        row_position[row]++;
    }

    free(row_counts);
    free(row_position);
}


/* Prodotto matrice-vettore utilizzando formato ELLPACK */
void matvec_ellpack(const int M, const int MAXNZ, const int *JA, const double *AS, const double *x, double *y) {
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
        for (int j = 0; j < MAXNZ; j++) {
            const int col = JA[i * MAXNZ + j];
            if (col != -1) {            // Controlla se l'indice di colonna è valido
                y[i] += AS[i * MAXNZ + j] * x[col];
            }
        }
    }
}
