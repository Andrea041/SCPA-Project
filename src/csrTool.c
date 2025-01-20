#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "../libs/csrTool.h"

/* Funzione per convertire la matrice in formato CSR */
void convert_to_csr(int M, int nz, const int *row_indices, const int *col_indices, const double *values, int **IRP, int **JA, double **AS) {
    *IRP = (int *)malloc((M + 1) * sizeof(int));    // Dimensione del vettore M
    *JA = (int *)malloc(nz * sizeof(int));          // Dimensione del vettore NZ - 1
    *AS = (double *)malloc(nz * sizeof(double));    // Dimensione del vettore NZ - 1

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

    int *row_position = malloc(M * sizeof(int));
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

/* Prodotto matrice-vettore serializzato */
void matvec_csr(int M, const int *IRP, const int *JA, const double *AS, double *x, double *y) {
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
        for (int j = IRP[i]; j < IRP[i + 1]; j++) {
            y[i] += AS[j] * x[JA[j]];
        }
    }

    // Scrittura dei risultati su file
    /*FILE *file = fopen("../result/risultati.txt", "w");
    if (file == NULL) {
        fprintf(stderr, "Errore nell'aprire il file.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < M; i++) {
        if (fprintf(file, "%.10f\n", y[i]) < 0) {
            fprintf(stderr, "Errore durante la scrittura nel file alla riga %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);*/
}

/* Prodotto matrice-vettore parallelo */
void matvec_csr_openMP(const int *IRP, const int *JA, const double *AS, const double *x, double *y, const int *start_row, const int *end_row, int num_threads, int nz, int M) {
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        for (int i = start_row[tid]; i < end_row[tid]; i++) {
            y[i] = 0.0;
            for (int j = IRP[i]; j < IRP[i + 1]; j++) {
                y[i] += AS[j] * x[JA[j]];
            }
        }
    }
}