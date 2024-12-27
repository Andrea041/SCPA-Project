#include <stdlib.h>
#include <stdio.h>

#include "../libs/csrTool.h"

/* Funzione per convertire la matrice in formato CSR */
void convert_to_csr(int M, int nz, const int *row_indices, const int *col_indices, const double *values, int **IRP, int **JA, double **AS) {
    *IRP = (int *)malloc((M + 1) * sizeof(int));
    *JA = (int *)malloc(nz * sizeof(int));
    *AS = (double *)malloc(nz * sizeof(double));

    if (*IRP == NULL || *JA == NULL || *AS == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        exit(1);
    }

    for (int i = 0; i <= M; i++) {
        (*IRP)[i] = 0;
    }

    for (int i = 0; i < nz; i++) {
        if (row_indices[i] < 0 || row_indices[i] >= M) {
            fprintf(stderr, "Errore: l'indice di riga è fuori dai limiti.\n");
            exit(1);
        }
        (*IRP)[row_indices[i] + 1]++;
    }

    for (int i = 0; i < M; i++) {
        (*IRP)[i + 1] += (*IRP)[i];
    }

    int *row_position = malloc(M * sizeof(int));  // Inizializza con 0
    if (row_position == NULL) {
        fprintf(stderr, "Errore nell'allocazione di row_position.\n");
        exit(1);
    }

    for (int i = 0; i < M; i++) {
        row_position[i] = 0;
    }

    for (int i = 0; i < nz; i++) {
        int row = row_indices[i];
        int pos = (*IRP)[row] + row_position[row];
        (*JA)[pos] = col_indices[i];
        (*AS)[pos] = values[i];
        row_position[row]++;
    }

    free(row_position);
}

/* Prodotto matrice-vettore */
void matvec_csr(int M, const int *IRP, const int *JA, const double *AS, const double *x, double *y) {
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
        for (int j = IRP[i]; j < IRP[i + 1]; j++) {
            y[i] += AS[j] * x[JA[j]];
        }
    }
}