#include <stdlib.h>
#include "../libs/hll_Operations.h"
#include "../libs/data_structure.h"
#include "../libs/hll_ellpack_Tool.h"

#include <omp.h>
#include <string.h>

#include "../libs/costants.h"

void convert_to_hll(int M, int N, int nz, const int *row_indices, const int *col_indices, const double *values, HLL_Matrix *hll_matrix) {
    int max_nz_per_row = hll_matrix->max_nz_per_row;

    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int start_row = block_idx * HackSize;
        int end_row = (block_idx + 1) * HackSize;
        if (end_row > M) end_row = M;

        int rows_in_block = end_row - start_row;
        int size = rows_in_block * max_nz_per_row;

        // Allocazione della memoria per il blocco
        hll_matrix->blocks[block_idx].JA = (int *)malloc(size * sizeof(int));
        hll_matrix->blocks[block_idx].AS = (double *)malloc(size * sizeof(double));

        if (!hll_matrix->blocks[block_idx].JA || !hll_matrix->blocks[block_idx].AS) {
            fprintf(stderr, "Errore: Allocazione fallita per il blocco %d.\n", block_idx);
            exit(EXIT_FAILURE);
        }

        // Inizializza la memoria allocata
        memset(hll_matrix->blocks[block_idx].JA, -1, size * sizeof(int));
        memset(hll_matrix->blocks[block_idx].AS, 0, size * sizeof(double));

        // Popola i dati
        for (int i = start_row; i < end_row; i++) {
            int pos = 0;
            int last_col_idx = 0; // Default: assegna 0 come indice valido

            for (int j = 0; j < nz; j++) {
                if (row_indices[j] == i) {
                    if (pos >= max_nz_per_row) {
                        fprintf(stderr, "Errore: Troppi elementi nella riga %d.\n", i);
                        exit(EXIT_FAILURE);
                    }
                    int index = pos + (i - start_row) * max_nz_per_row;

                    hll_matrix->blocks[block_idx].JA[index] = col_indices[j]; // Base 0
                    hll_matrix->blocks[block_idx].AS[index] = values[j];
                    last_col_idx = col_indices[j]; // Aggiorna l'ultimo indice valido

                    // Debug: stampa l'inserimento
                    printf("Scrivendo: JA[%d] = %d, AS[%d] = %.10f (Riga %d)\n",
                           index, hll_matrix->blocks[block_idx].JA[index],
                           index, hll_matrix->blocks[block_idx].AS[index], i);

                    pos++;
                }
            }

            // Verifica se `last_col_idx` è stato aggiornato
            if (last_col_idx < 0 || last_col_idx >= N) {
                last_col_idx = 0; // Imposta un valore di fallback valido
            }

            // Riempi gli spazi vuoti con valori di default
            while (pos < max_nz_per_row || pos>=M) {
                int index = pos + (i - start_row) * max_nz_per_row;
                hll_matrix->blocks[block_idx].JA[index] = last_col_idx; // Usa l'ultimo indice valido
                hll_matrix->blocks[block_idx].AS[index] = 0.0;          // Valore nullo
                pos++;

                // Debug: stampa il riempimento
                printf("Riempimento: JA[%d] = %d, AS[%d] = %.10f (Riga %d)\n",
                       index, hll_matrix->blocks[block_idx].JA[index],
                       index, hll_matrix->blocks[block_idx].AS[index], i);
            }
        }
    }
}


// Funzione per il prodotto matrice-vettore in formato HLL
void matvec_Hll(HLL_Matrix *hll_matrix, double *x, double *y, int num_threads, int *start_row, int *end_row, int N, int M) {
    // Parallelizza il calcolo del prodotto matrice-vettore
#pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int start = start_row[thread_id];
        int end = end_row[thread_id];

        for (int block_idx = start / HackSize; block_idx <= end / HackSize; block_idx++) {
            int block_start_row = block_idx * HackSize;
            int block_end_row = (block_idx + 1) * HackSize;
            if (block_end_row > M) block_end_row = M;

            ELLPACK_Block *block = &hll_matrix->blocks[block_idx];

            for (int row_idx = block_start_row; row_idx < block_end_row; row_idx++) {
                if (row_idx < start || row_idx > end) continue;

                double temp_sum = 0.0;

                for (int j = 0; j < hll_matrix->max_nz_per_row; j++) {
                    int col_idx = block->JA[(row_idx - block_start_row) * hll_matrix->max_nz_per_row + j];
                    double value = block->AS[(row_idx - block_start_row) * hll_matrix->max_nz_per_row + j];

                    if (col_idx >= 0) { // Colonna valida
                        temp_sum += value * x[col_idx];
                    }
                }

                y[row_idx] += temp_sum;
            }
        }
    }

    // Debug: Stampa il risultato ordinato
    printf("Risultato del prodotto matrice-vettore:\n");
    for (int i = 0; i < M; i++) {
        printf("y[%d] = %.10f\n", i, y[i]);
    }
}

