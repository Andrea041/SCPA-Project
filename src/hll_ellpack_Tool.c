#include <stdlib.h>
#include "../libs/hll_Operations.h"
#include "../libs/data_structure.h"
#include "../libs/hll_ellpack_Tool.h"
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <tgmath.h>

#include "../libs/costants.h"

// Funzione per convertire una matrice sparsa al formato HLL
// Funzione per convertire una matrice sparsa al formato HLL
void convert_to_hll(int M, int N, int nz, const int *row_indices, const int *col_indices, const double *values, HLL_Matrix *hll_matrix) {
    int max_nz_per_row = hll_matrix->max_nz_per_row;

    int *row_start = calloc(M + 1, sizeof(int));
    if (!row_start) {
        fprintf(stderr, "Errore: allocazione memoria fallita per row_start.\n");
        exit(EXIT_FAILURE);
    }

    // Calcola gli indici di inizio per ciascuna riga
    for (int i = 0; i < nz; i++) {
        row_start[row_indices[i] + 1]++;
    }
    for (int i = 1; i <= M; i++) {
        row_start[i] += row_start[i - 1];
    }

    int *sorted_col_indices = malloc(nz * sizeof(int));
    double *sorted_values = malloc(nz * sizeof(double));

    if (!sorted_col_indices || !sorted_values) {
        fprintf(stderr, "Errore: allocazione memoria fallita per array ordinati.\n");
        free(row_start);
        exit(EXIT_FAILURE);
    }

    // Ordinamento delle colonne e dei valori
    for (int i = 0; i < nz; i++) {
        int row = row_indices[i];
        int pos = row_start[row]++;
        sorted_col_indices[pos] = col_indices[i];
        sorted_values[pos] = values[i];
    }

    // Ripristina gli indici di inizio delle righe
    for (int i = M; i > 0; i--) {
        row_start[i] = row_start[i - 1];
    }
    row_start[0] = 0;

    // Elabora i blocchi
    #pragma omp parallel for
    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int start_row = block_idx * HackSize;
        int end_row = (block_idx + 1) * HackSize;
        if (end_row > M) end_row = M;

        int rows_in_block = end_row - start_row;
        int size_of_arrays = max_nz_per_row * rows_in_block;

        // Inizializza la memoria per il blocco
        hll_matrix->blocks[block_idx].JA = (int *)malloc(size_of_arrays * sizeof(int));
        hll_matrix->blocks[block_idx].AS = (double *)malloc(size_of_arrays * sizeof(double));

        if (!hll_matrix->blocks[block_idx].JA || !hll_matrix->blocks[block_idx].AS) {
            fprintf(stderr, "Errore: Allocazione fallita per il blocco %d.\n", block_idx);
            exit(EXIT_FAILURE);
        }

        // Elabora le righe del blocco
        for (int i = start_row; i < end_row; i++) {
            int row_offset = (i - start_row) * max_nz_per_row;
            int row_nz_start = row_start[i];
            int row_nz_end = row_start[i + 1];

            int pos = 0;
            int last_col_idx = -1;

            // Gestione degli elementi non-zero per la riga
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
            int index = row_offset + pos;
            pos=index;
            // Padding per righe con meno di max_nz_per_row
            while (pos < max_nz_per_row) {

                hll_matrix->blocks[block_idx].JA[index] = last_col_idx;
                hll_matrix->blocks[block_idx].AS[index] = 0.0;
                pos++;
            }
        }

        printf("Blocco %d completato\n", block_idx + 1);
    }

    free(row_start);
    free(sorted_col_indices);
    free(sorted_values);
    printf("Conversione HLL completata!\n");
}





void matvec_Hll(HLL_Matrix *hll_matrix, double *x, double *y, int num_threads, int *start_row, int *end_row, int N, int M) {
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();

        for (int block_idx = start_row[thread_id] / HackSize; block_idx <= end_row[thread_id] / HackSize; block_idx++) {
            int block_start_row = block_idx * HackSize;
            int block_end_row = (block_idx + 1) * HackSize;
            if (block_end_row > M) block_end_row = M;

            ELLPACK_Block *block = &hll_matrix->blocks[block_idx];

            for (int row_idx = block_start_row; row_idx < block_end_row; row_idx++) {
                if (row_idx < start_row[thread_id] || row_idx > end_row[thread_id]) continue;

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
    //debug per verificare che hll funzionasse
/*
    FILE *file = fopen("../result/risultati.txt", "r");
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
