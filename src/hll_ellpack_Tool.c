#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
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

void convert_to_hll(struct matrixData *matrix_data, HLL_Matrix *hll_matrix) {
    int *row_start = malloc((matrix_data->M + 1) * sizeof(int));
    if (!row_start) {
        fprintf(stderr, "Errore: allocazione memoria fallita per row_start.\n");
        exit(EXIT_FAILURE);
    }

    memset(row_start, 0, (matrix_data->M + 1) * sizeof(int));  // Inizializzazione a 0

    // Conta gli elementi in ogni riga
    for (int i = 0; i < matrix_data->nz; i++) {
        row_start[matrix_data->row_indices[i] + 1]++;
    }

    // Calcola gli offset cumulativi
    for (int i = 1; i <= matrix_data->M; i++) {
        row_start[i] += row_start[i - 1];
    }

    int *sorted_col_indices = malloc(matrix_data->nz * sizeof(int));
    double *sorted_values = malloc(matrix_data->nz * sizeof(double));
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

    int *nz_per_row = calloc(matrix_data->M, sizeof(int));
    if (!nz_per_row) {
        fprintf(stderr, "Errore: Allocazione fallita per nz_per_row.\n");
        exit(EXIT_FAILURE);
    }

    calculate_max_nz_in_row_in_block(matrix_data, nz_per_row);

    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int start_row = block_idx * HackSize;
        int end_row = (block_idx + 1) * HackSize;
        if (end_row > matrix_data->M) end_row = matrix_data->M;

        hll_matrix->blocks[block_idx].max_nz_per_row = find_max_nz(nz_per_row, start_row, end_row);

        int max_nz_per_row = hll_matrix->blocks[block_idx].max_nz_per_row;

        int rows_in_block = end_row - start_row;
        int size_of_arrays = max_nz_per_row * rows_in_block;

        hll_matrix->blocks[block_idx].JA = (int *)malloc(size_of_arrays * sizeof(int));
        hll_matrix->blocks[block_idx].AS = (double *)malloc(size_of_arrays * sizeof(double));
        if (!hll_matrix->blocks[block_idx].JA || !hll_matrix->blocks[block_idx].AS) {
            fprintf(stderr, "Errore: allocazione memoria fallita per il blocco %d.\n", block_idx);
            free(row_start);
            free(sorted_col_indices);
            free(sorted_values);
            exit(EXIT_FAILURE);
        }

        memset(hll_matrix->blocks[block_idx].JA, -1, size_of_arrays * sizeof(int));  // Default
        memset(hll_matrix->blocks[block_idx].AS, 0, size_of_arrays * sizeof(double));

        // Copia i valori per riga
        for (int i = start_row; i < end_row; i++) {
            int row_offset = (i - start_row) * max_nz_per_row;
            int row_nz_start = row_start[i];
            int row_nz_end = row_start[i + 1];
            for (int j = row_nz_start, pos = 0; j < row_nz_end; j++, pos++) {
                hll_matrix->blocks[block_idx].JA[row_offset + pos] = sorted_col_indices[j];
                hll_matrix->blocks[block_idx].AS[row_offset + pos] = sorted_values[j];
            }
        }
    }

    free(row_start);
    free(sorted_col_indices);
    free(sorted_values);
}

void matvec_Hll(HLL_Matrix *hll_matrix, double *x, double *y, int num_threads, int M) {
  #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();

        // Definiamo i limiti delle righe per ogni thread
        int start_row = M / num_threads * thread_id;
        int end_row = thread_id == num_threads - 1 ? M : M / num_threads * (thread_id + 1);

        for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
            int block_start_row = block_idx * HackSize;
            int block_end_row = (block_idx + 1) * HackSize;
            if (block_end_row > M) block_end_row = M;

            int max_nz_per_row = hll_matrix->blocks[block_idx].max_nz_per_row;
            ELLPACK_Block *block = &hll_matrix->blocks[block_idx];

            // Itera sulle righe del blocco, solo quelle assegnate al thread
            for (int row_idx = block_start_row; row_idx < block_end_row; row_idx++) {
                if (row_idx < start_row || row_idx >= end_row) continue;

                double temp_sum = 0.0;

                // Itera sui non-zero per ogni riga
                for (int j = 0; j < max_nz_per_row; j++) {
                    int col_idx = block->JA[(row_idx - block_start_row) * max_nz_per_row + j];
                    double value = block->AS[(row_idx - block_start_row) * max_nz_per_row + j];

                    if (col_idx >= 0) { // Colonna valida
                        temp_sum += value * x[col_idx];
                    }
                }

                // Aggiungi il risultato al vettore y
                y[row_idx] += temp_sum;
            }
        }
    }
    //debug per verificare che hll funzionasse

   /*FILE *file = fopen("../result/risultati.txt", "r");
    if (file == NULL) {
        perror("Errore nell'aprire il file");
        exit(EXIT_FAILURE);
    }

    for (int t = 0; t < M; t++) {
        double file_value;
        if (fscanf(file, "%lf", &file_value) != 1) {
            fprintf(stderr, "Errore nella lettura del file. Non sono stati letti abbastanza valori.\n");
            exit(EXIT_FAILURE);
        }

        // Confronta il valore letto dal file con il valore calcolato
        if (fabs(file_value - y[t]) > 1e-10) { // Usa una tolleranza per confrontare i valori a causa di errori di precisione
            fprintf(stderr, "Errore: Il valore di y[%d] calcolato (%.10f) non corrisponde al valore nel file (%.10f).\n", t, y[t], file_value);
            exit(EXIT_FAILURE);
        }

    }

    fclose(file); // Chiude il file
    printf("Controllo completato, tutti i valori di y sono corretti.\n");*/
}
