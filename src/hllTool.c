#include <stdlib.h>

#include "../libs/hllTool.h"

// Funzione per convertire la matrice in formato HLL
void convert_to_ellpack(int M, int N, int nz, int *row_indices, int *col_indices, double *values, int **JA, double **AS, int *MAXNZ) {
    *MAXNZ = 0;

    // Trova il massimo numero di non-zeri per riga
    int *row_counts = (int *)calloc(M, sizeof(int));
    for (int i = 0; i < nz; i++) {
        row_counts[row_indices[i]]++;
    }
    for (int i = 0; i < M; i++) {
        if (row_counts[i] > *MAXNZ) {
            *MAXNZ = row_counts[i];
        }
    }

    // Alloca memoria per JA e AS
    *JA = (int *)malloc(M * (*MAXNZ) * sizeof(int));
    *AS = (double *)malloc(M * (*MAXNZ) * sizeof(double));

    // Inizializza JA e AS con valori di default
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < *MAXNZ; j++) {
            (*JA)[i * (*MAXNZ) + j] = (j == 0) ? 1 : (*JA)[i * (*MAXNZ) + j - 1]; // Indice valido o precedente
            (*AS)[i * (*MAXNZ) + j] = 0.0; // Valore zero per celle vuote
        }
    }

    // Popola JA e AS
    int *row_position = (int *)calloc(M, sizeof(int));
    for (int i = 0; i < nz; i++) {
        int row = row_indices[i];
        int pos = row_position[row];
        (*JA)[row * (*MAXNZ) + pos] = col_indices[i] + 1; // Indici 1-based
        (*AS)[row * (*MAXNZ) + pos] = values[i];
        row_position[row]++;
    }

    free(row_counts);
    free(row_position);
}

// Prodotto matrice-vettore utilizzando formato ELLPACK
void matvec_ellpack(int M, int MAXNZ, int *JA, double *AS, double *x, double *y) {
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
        for (int j = 0; j < MAXNZ; j++) {
            int col = JA[i * MAXNZ + j];
            if (col != -1) { // Controlla se l'indice di colonna è valido
                y[i] += AS[i * MAXNZ + j] * x[col];
            }
        }
    }
}
