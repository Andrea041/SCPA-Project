#include <stdlib.h>

#include "../libs/csrTool.h"

// Funzione per convertire la matrice
void convert_to_csr(int M, int N, int nz, int *row_indices, int *col_indices, double *values, int **IRP, int **JA, double **AS) {
    *IRP = (int *)malloc((M + 1) * sizeof(int)); // Array dei puntatori di riga
    *JA = (int *)malloc(nz * sizeof(int));       // Array degli indici di colonna
    *AS = (double *)malloc(nz * sizeof(double)); // Array dei valori non nulli

    // Inizializza IRP a zero
    for (int i = 0; i <= M; i++) {
        (*IRP)[i] = 0;
    }

    // Conta i non-zero per ogni riga
    for (int i = 0; i < nz; i++) {
        (*IRP)[row_indices[i] + 1]++;
    }

    // Costruisci IRP come somma cumulativa
    for (int i = 0; i < M; i++) {
        (*IRP)[i + 1] += (*IRP)[i];
    }

    // Popola JA e AS
    int *row_position = (int *)calloc(M, sizeof(int));
    for (int i = 0; i < nz; i++) {
        int row = row_indices[i];
        int pos = (*IRP)[row] + row_position[row];
        (*JA)[pos] = col_indices[i];
        (*AS)[pos] = values[i];
        row_position[row]++;
    }

    free(row_position);
}

// Prodotto matrice-vettore
void matvec_csr(int M, int *IRP, int *JA, double *AS, double *x, double *y) {
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
        for (int j = IRP[i]; j < IRP[i + 1]; j++) {
            y[i] += AS[j] * x[JA[j]];
        }
    }
}