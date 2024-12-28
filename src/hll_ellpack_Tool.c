#include <stdlib.h>

#include "../libs/data_structure.h"
#include "../libs/hll_ellpack_Tool.h"

/* Funzione Hll per convertire la matrice in formato ELLPACK */
void convert_to_hll(int M, int N, int nz, const int *row_indices, const int *col_indices, const double *values, HLL_Matrix *hll_matrix) {
    int max_nz_per_row = 0;  // Trovare il numero massimo di non nulli per riga

    // Passo 1: Troviamo il numero massimo di non nulli per riga
    for (int i = 0; i < M; i++) {
        int count = 0;
        for (int j = 0; j < nz; j++) {
            if (row_indices[j] == i) {
                count++;
            }
        }
        if (count > max_nz_per_row) {
            max_nz_per_row = count;
        }
    }

    hll_matrix->max_nz_per_row = max_nz_per_row;
    hll_matrix->num_blocks = (M + HackSize - 1) / HackSize;  // Numero di blocchi
    hll_matrix->blocks = (ELLPACK_Block *)malloc(hll_matrix->num_blocks * sizeof(ELLPACK_Block));

    // Passo 2: Partizioniamo la matrice in blocchi e li memorizziamo in formato ELLPACK
    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int start_row = block_idx * HackSize;
        int end_row = (block_idx + 1) * HackSize;
        if (end_row > M) end_row = M;

        // Inizializza gli array per il blocco
        hll_matrix->blocks[block_idx].JA = (int *)malloc(HackSize * max_nz_per_row * sizeof(int));
        hll_matrix->blocks[block_idx].AS = (double *)malloc(HackSize * max_nz_per_row * sizeof(double));

        // Inizializza a 0 gli indici di colonna e i valori
        for (int i = 0; i < HackSize * max_nz_per_row; i++) {
            hll_matrix->blocks[block_idx].JA[i] = 0;
            hll_matrix->blocks[block_idx].AS[i] = 0.0;
        }

        // Copia i dati relativi al blocco
        for (int i = start_row; i < end_row; i++) {
            int pos = 0;
            int last_col_idx = 0;  // Variabile per tracciare l'ultimo indice di colonna valido
            for (int j = 0; j < nz; j++) {
                if (row_indices[j] == i) {
                    hll_matrix->blocks[block_idx].JA[pos] = col_indices[j] + 1;  // Indici a base 1
                    hll_matrix->blocks[block_idx].AS[pos] = values[j];
                    last_col_idx = col_indices[j] + 1;  // Aggiorna l'ultimo indice valido
                    pos++;
                }
            }

            // Riempie gli spazi rimanenti con l'ultimo indice valido
            while (pos < max_nz_per_row) {
                hll_matrix->blocks[block_idx].JA[pos] = last_col_idx;
                hll_matrix->blocks[block_idx].AS[pos] = 0.0;  // Valore zero per il non-nullo mancante
                pos++;
            }
        }
    }
}

void matvec_Hll(HLL_Matrix *hll_matrix, double *x, double *y) {
    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        ELLPACK_Block block = hll_matrix->blocks[block_idx];

        for (int i = 0; i < HackSize; i++) {
            double t = 0.0;
            for (int j = 0; j < hll_matrix->max_nz_per_row; j++) {
                if (block.JA[i * hll_matrix->max_nz_per_row + j] != 0) {
                    t += block.AS[i * hll_matrix->max_nz_per_row + j] * x[block.JA[i * hll_matrix->max_nz_per_row + j] - 1];  // Indici a base 1
                }
            }
            y[i + block_idx * HackSize] = t;
        }
    }
}

