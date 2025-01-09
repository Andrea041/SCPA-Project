#include <stdlib.h>
#include "../libs/hll_Operations.h"
#include "../libs/data_structure.h"
#include "../libs/hll_ellpack_Tool.h"

#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <tgmath.h>

#include "../libs/costants.h"


// Funzione per calcolare il massimo numero di non-zero nella riga contenuta nel blocco
#include <omp.h>

void calculate_max_nz_in_row_in_block(const struct matrixData *matrix_data, int block_idx, int start_row, int end_row, HLL_Matrix *hll_matrix) {
    // Array temporaneo per contare i non-zero per riga nel blocco
    int rows_in_block = end_row - start_row;
    int *nz_per_row = calloc(rows_in_block, sizeof(int));
    if (!nz_per_row) {
        fprintf(stderr, "Errore: Allocazione fallita per nz_per_row.\n");
        exit(EXIT_FAILURE);
    }

    // Conta i non-zero per ogni riga nel blocco in parallelo
#pragma omp parallel for
    for (int i = 0; i < matrix_data->nz; i++) {
        int row_idx = matrix_data->row_indices[i];

        // Verifica se la riga appartiene al blocco
        if (row_idx >= start_row && row_idx < end_row) {
            int local_row = row_idx - start_row;

            // Incremento protetto da race condition
#pragma omp atomic
            nz_per_row[local_row]++;
        }
    }

    // Trova il massimo numero di non-zero nella riga contenuta nel blocco in parallelo
    int max_nz_in_row = 0;

#pragma omp parallel for reduction(max : max_nz_in_row)
    for (int i = 0; i < rows_in_block; i++) {
        if (nz_per_row[i] > max_nz_in_row) {
            max_nz_in_row = nz_per_row[i];
        }
    }

    // Memorizza il massimo nella struttura ELLPACK del blocco corrente
    hll_matrix->blocks[block_idx].max_nz_per_row = max_nz_in_row;

    // Libera la memoria temporanea
    free(nz_per_row);
}





// Funzione per convertire una matrice sparsa al formato HLL
void convert_to_hll(int M, int N, int nz, const int *row_indices, const int *col_indices, const double *values, HLL_Matrix *hll_matrix) {
    // Pre-calcola la mappa degli elementi per riga
    int *row_start = calloc(M + 1, sizeof(int));
    if (!row_start) {
        fprintf(stderr, "Errore: allocazione memoria fallita per row_start.\n");
        exit(EXIT_FAILURE);
    }

    // Conta gli elementi in ogni riga
    for (int i = 0; i < nz; i++) {
        row_start[row_indices[i] + 1]++;
    }

    // Calcola gli offset
    for (int i = 1; i <= M; i++) {
        row_start[i] += row_start[i - 1];
    }

    // Prepara array ordinati per indici di colonna e valori
    int *sorted_col_indices = malloc(nz * sizeof(int));
    double *sorted_values = malloc(nz * sizeof(double));
    if (!sorted_col_indices || !sorted_values) {
        fprintf(stderr, "Errore: allocazione memoria fallita per array ordinati.\n");
        free(row_start);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < nz; i++) {
        int row = row_indices[i];
        int pos = row_start[row]++;
        sorted_col_indices[pos] = col_indices[i];
        sorted_values[pos] = values[i];
    }

    // Ripristina row_start
    for (int i = M; i > 0; i--) {
        row_start[i] = row_start[i - 1];
    }
    row_start[0] = 0;

    // Parallelizza il lavoro tra i blocchi
    #pragma omp parallel for
    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int start_row = block_idx * HackSize;
        int end_row = (block_idx + 1) * HackSize;
        if (end_row > M) end_row = M;

        // Calcola il massimo numero di non-zero per riga in questo blocco
        calculate_max_nz_in_row_in_block(&(struct matrixData){.row_indices = row_indices, .col_indices = col_indices, .values = values, .M = M, .N = N, .nz = nz}, block_idx, start_row, end_row, hll_matrix);

        // Usa il massimo calcolato per allocare memoria e popolare i dati
        int max_nz_per_row = hll_matrix->blocks[block_idx].max_nz_per_row;
        int rows_in_block = end_row - start_row;
        int size_of_arrays = max_nz_per_row * rows_in_block;

        // Allocazione della memoria per il blocco
        hll_matrix->blocks[block_idx].JA = (int *)malloc(size_of_arrays * sizeof(int));
        hll_matrix->blocks[block_idx].AS = (double *)malloc(size_of_arrays * sizeof(double));
        if (!hll_matrix->blocks[block_idx].JA || !hll_matrix->blocks[block_idx].AS) {
            fprintf(stderr, "Errore: allocazione memoria fallita per il blocco %d.\n", block_idx);
            free(row_start);
            free(sorted_col_indices);
            free(sorted_values);
            exit(EXIT_FAILURE);
        }

        // Popola i dati
        for (int i = start_row; i < end_row; i++) {
            int row_offset = (i - start_row) * max_nz_per_row;
            int row_nz_start = row_start[i];
            int row_nz_end = row_start[i + 1];
            int pos = 0;
            int last_col_idx = -1;
            for (int j = row_nz_start; j < row_nz_end; j++) {
                if (pos >= max_nz_per_row) {
                    fprintf(stderr, "Errore: Troppi elementi nella riga %d.\n", i);
                    exit(EXIT_FAILURE);
                }
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = sorted_col_indices[j];
                hll_matrix->blocks[block_idx].AS[index] = sorted_values[j];
                last_col_idx = sorted_col_indices[j];
                pos++;
            }

            // Riempi gli spazi vuoti con valori di default
            while (pos < max_nz_per_row) {
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = last_col_idx;
                hll_matrix->blocks[block_idx].AS[index] = 0.0;
                pos++;
            }
        }

        printf("Blocco %d completato\n", block_idx);
    }

    // Libera memoria temporanea
    free(row_start);
    free(sorted_col_indices);
    free(sorted_values);
}


void matvec_Hll(HLL_Matrix *hll_matrix, double *x, double *y, int num_threads, int *start_row, int *end_row, int N, int M) {
  #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();

        // Definiamo i limiti delle righe per ogni thread
        int start_row = (M / num_threads) * thread_id;
        int end_row = (thread_id == num_threads - 1) ? M : (M / num_threads) * (thread_id + 1);

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
   /* for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int start_row_block = block_idx * HackSize;
        int end_row_block = (block_idx + 1) * HackSize;
        if (end_row_block > M) end_row_block = M;

        int rows_in_block = end_row_block - start_row_block;
        int max_nz_per_row = hll_matrix->blocks[block_idx].max_nz_per_row;

        for (int i = 0; i < rows_in_block; i++) {
            int row_idx = start_row_block + i;
            int row_offset = i * max_nz_per_row;

            for (int j = 0; j < max_nz_per_row; j++) {
                int col_idx = hll_matrix->blocks[block_idx].JA[row_offset + j];
                double value = hll_matrix->blocks[block_idx].AS[row_offset + j];

                if (value != 0) {
                    y[row_idx] += value * x[col_idx];
                }
            }
        }
    }*/
    //debug per verificare che hll funzionasse

   /* FILE *file = fopen("../result/risultati.txt", "r");
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
