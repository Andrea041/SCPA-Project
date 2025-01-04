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
    // Calcolo del prodotto matrice-vettore
    for (int i = 0; i < M; i++) {
        y[i] = 0.0; // Inizializza y[i] a 0
        for (int j = IRP[i]; j < IRP[i + 1]; j++) {
            y[i] += AS[j] * x[JA[j]]; // Prodotto scalare per riga
        }
    }
}


/*void matvec_csr_openMP(const int *IRP, const int *JA, const double *AS, const double *x, double *y, int** thread_rows, const int *row_counts, int num_threads, const int M) {
    if (num_threads == 1) {
        for (int i = 0; i < M; i++) {
            y[i] = 0.0; // Inizializza y[i] a 0
            for (int j = IRP[i]; j < IRP[i + 1]; j++) {
                y[i] += AS[j] * x[JA[j]]; // Prodotto scalare per riga
            }
        }
    }else {
#pragma omp parallel num_threads(2)
        {
            // Ottieni l'ID del thread corrente
            int thread_id = omp_get_thread_num();

            // Ottieni le righe assegnate a questo thread
            int* rows = thread_rows[thread_id];  // Array delle righe assegnate al thread
            int num_rows = row_counts[thread_id]; // Numero di righe assegnate al thread

            // Calcola i limiti di inizio e fine
            int start = rows[0];               // Prima riga assegnata al thread
            int end = rows[num_rows - 1] + 1;  // Ultima riga assegnata al thread (+1 per includere l'ultima riga)

            // Itera direttamente da start a end
            for (int row = start; row < end; row++) {
                y[row] = 0.0;      // Inizializza il valore per la riga corrente

                // Scorri gli elementi non nulli della riga nella matrice CSR
                for (int j = IRP[row]; j < IRP[row + 1]; j++) {
                    y[row] += AS[j] * x[JA[j]]; // Prodotto scalare della riga con il vettore x
                }
            }
        }

    }
}*/

void matvec_csr_openMP(const int *IRP, const int *JA, const double *AS, const double *x, double *y, int *start_row, int *end_row, int num_threads, int nz, int M) {
    if (nz < 10000) {
        for (int i = 0; i < M; i++) {
            y[i] = 0.0; // Inizializza y[i] a 0
            for (int j = IRP[i]; j < IRP[i + 1]; j++) {
                y[i] += AS[j] * x[JA[j]]; // Prodotto scalare per riga
            }
        }
    } else {
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
}