#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../libs/data_structure.h"
#include "../libs/hll_ellpack_Tool.h"
#include "../libs/costants.h"

/* Funzione per calcolare il massimo numero di nonzeri per ciascuna riga */
void calculate_max_nz_in_row_in_block(const struct matrixData *matrix_data, int *nz_per_row) {
    for (int i = 0; i < matrix_data->nz; i++) {
        int row_idx = matrix_data->row_indices[i];
        nz_per_row[row_idx]++;
    }
}

/* Funzione per trovare il massimo numero di nonzeri all'interno di un intervallo di righe */
int find_max_nz(const int *nz_per_row, int start_row, int end_row) {
    int max_nz = 0;
    for (int i = start_row; i < end_row; i++) {
        if (nz_per_row[i] > max_nz)
            max_nz = nz_per_row[i];
    }
    return max_nz;
}

int find_max_nz_per_block(const int *nz_per_row, int start_row, int end_row) {
    int tot_nz = 0;
    for (int i = start_row; i < end_row; i++) {
        tot_nz += nz_per_row[i];
    }
    return tot_nz;
}

/* Funzione per convertire una matrice in formato HLL su CPU */
void convert_to_hll_cuda( matrixData *matrix_data, HLL_Matrix *hll_matrix) {
    int *row_start = (int *)calloc(matrix_data->M + 1, sizeof(int));
    if (!row_start) {
        fprintf(stderr, "Errore: allocazione memoria fallita per row_start.\n");
        exit(EXIT_FAILURE);
    }

    // Conta gli elementi in ogni riga
    for (int i = 0; i < matrix_data->nz; i++) {
        row_start[matrix_data->row_indices[i] + 1]++;
    }

    // Calcola gli offset cumulativi
    for (int i = 1; i <= matrix_data->M; i++) {
        row_start[i] += row_start[i - 1];
    }

    int *sorted_col_indices = (int *)malloc(matrix_data->nz * sizeof(int));
    double *sorted_values = (double *)malloc(matrix_data->nz * sizeof(double));
    if (!sorted_col_indices || !sorted_values) {
        fprintf(stderr, "Errore: allocazione memoria fallita per array ordinati.\n");
        free(row_start);
        exit(EXIT_FAILURE);
    }

    // Ordina i dati per riga
    for (int i = 0; i < matrix_data->nz; i++) {
        int row = matrix_data->row_indices[i];
        int pos = row_start[row]++;
        sorted_col_indices[pos] = matrix_data->col_indices[i];
        sorted_values[pos] = matrix_data->values[i];
    }

    // Ripristina row_start
    for (int i = matrix_data->M; i > 0; i--) {
        row_start[i] = row_start[i - 1];
    }
    row_start[0] = 0;

    int *nz_per_row = (int *)calloc(matrix_data->M, sizeof(int));
    if (!nz_per_row) {
        fprintf(stderr, "Errore: Allocazione fallita per nz_per_row.\n");
        free(row_start);
        free(sorted_col_indices);
        free(sorted_values);
        exit(EXIT_FAILURE);
    }

    calculate_max_nz_in_row_in_block(matrix_data, nz_per_row);

    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int start_row = block_idx * HackSize;
        int end_row = (block_idx + 1) * HackSize;
        if (end_row > matrix_data->M) end_row = matrix_data->M;

        hll_matrix->blocks[block_idx].nz_per_block = find_max_nz_per_block(nz_per_row, start_row, end_row);
        hll_matrix->blocks[block_idx].max_nz_per_row = find_max_nz(nz_per_row, start_row, end_row);

        int max_nz_per_row = hll_matrix->blocks[block_idx].max_nz_per_row;
        int rows_in_block = end_row - start_row;
        int size_of_arrays = max_nz_per_row * rows_in_block;
        if (max_nz_per_row < 0 || rows_in_block < 0) {
            fprintf(stderr, "Errore: Valori invalidi per il blocco %d: %d - %d\n", block_idx, rows_in_block, max_nz_per_row);
            free(row_start);
            free(sorted_col_indices);
            free(sorted_values);
            free(nz_per_row);
            exit(EXIT_FAILURE);
        }

        hll_matrix->blocks[block_idx].JA = (int *)calloc(size_of_arrays, sizeof(int));
        hll_matrix->blocks[block_idx].AS = (double *)calloc(size_of_arrays, sizeof(double));
        if (!hll_matrix->blocks[block_idx].JA || !hll_matrix->blocks[block_idx].AS) {
            fprintf(stderr, "Errore: allocazione memoria fallita per il blocco %d.\n", block_idx);
            for (int k = 0; k <= block_idx; k++) {
                free(hll_matrix->blocks[k].JA);
                free(hll_matrix->blocks[k].AS);
            }
            free(row_start);
            free(sorted_col_indices);
            free(sorted_values);
            free(nz_per_row);
            exit(EXIT_FAILURE);
        }

        memset(hll_matrix->blocks[block_idx].JA, -1, size_of_arrays * sizeof(int));
        memset(hll_matrix->blocks[block_idx].AS, 0, size_of_arrays * sizeof(double));

        for (int i = start_row; i < end_row; i++) {
            int row_offset = (i - start_row) * max_nz_per_row;
            int row_nz_start = row_start[i];
            int row_nz_end = row_start[i + 1];

            int pos = 0;
            int last_col_idx = -1;

            for (int j = row_nz_start; j < row_nz_end; j++) {
                if (pos >= max_nz_per_row) break;
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = sorted_col_indices[j];
                hll_matrix->blocks[block_idx].AS[index] = sorted_values[j];
                last_col_idx = sorted_col_indices[j];
                pos++;
            }

            while (pos < max_nz_per_row) {
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = last_col_idx;
                hll_matrix->blocks[block_idx].AS[index] = 0.0;
                pos++;
            }
        }
    }

    free(row_start);
    free(sorted_col_indices);
    free(sorted_values);
    free(nz_per_row);
}



/*__global__ void matvec_Hll_cuda(HLL_Matrix *d_hll_matrix, double *d_x, double *d_y, int M) {
    for (int block_id = 0; block_id < d_hll_matrix->num_blocks; block_id++) {

        int start_row = block_id * HackSize;
        int end_row = (block_id + 1) * HackSize;
        if (end_row > M) end_row = M;

        int row_offset = 0;

        for (int i = start_row; i < end_row; i++) {
            d_y[i] = 0.0;

            for (int j = 0; j < d_hll_matrix->blocks[block_id].max_nz_per_row; j++) {
                d_y[i] += d_hll_matrix->blocks[block_id].AS[j + row_offset] * d_x[d_hll_matrix->blocks[block_id].JA[j + row_offset]];
            }

            row_offset += d_hll_matrix->blocks[block_id].max_nz_per_row;
        }
    }
}*/

__global__ void matvec_Hll_cuda(const HLL_Matrix *d_hll_matrix, const double *d_x, double *d_y, int M) {
    // Calcola l'indice della riga globale
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;

    // Controlla che l'indice non ecceda il numero di righe
    if (global_row >= M) return;

    // Trova il blocco corrispondente
    int block_id = global_row / HackSize;
    int local_row = global_row % HackSize;

    // Ottieni il blocco corrente
    const ELLPACK_Block *block = &d_hll_matrix->blocks[block_id];

    // Calcola l'offset della riga all'interno del blocco
    int row_offset = local_row * block->max_nz_per_row;

    // Calcola il prodotto matrice-vettore
    double result = 0.0;
    for (int j = 0; j < block->max_nz_per_row; j++) {
        int col = block->JA[row_offset + j];
        double value = block->AS[row_offset + j];
        result += value * d_x[col];
    }

    // Salva il risultato
    d_y[global_row] = result;
}
