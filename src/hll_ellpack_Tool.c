#include <stdlib.h>
#include "../libs/hll_Operations.h"
#include "../libs/data_structure.h"
#include "../libs/hll_ellpack_Tool.h"

#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <tgmath.h>

#include "../libs/costants.h"

// Funzione per calcolare il massimo numero di non-zero per riga nel blocco
void calculate_max_nz_per_row_in_block(int M, const int *row_indices, int nz, int block_idx, int start_row, int end_row, HLL_Matrix *hll_matrix) {
    // Variabile per calcolare il massimo numero di non-zero per riga nel blocco
    int max_nz_per_row_in_block = 0;

    // Parallelizzazione del conteggio dei non-zero per riga
    //#pragma omp parallel for reduction(max:max_nz_per_row_in_block)
    for (int i = 0; i < nz; i++) {
        int row_idx = row_indices[i];

        // Verifica se la riga è nel range del blocco
        if (row_idx >= start_row && row_idx < end_row) {
            max_nz_per_row_in_block++;
        }
    }

    // Memorizza il valore massimo nel blocco corrente della matrice HLL
    hll_matrix->blocks[block_idx].max_nz_per_row = max_nz_per_row_in_block;
}



// Funzione per convertire una matrice sparsa al formato HLL
void convert_to_hll(int M, int N, int nz, const int *row_indices, const int *col_indices, const double *values, HLL_Matrix *hll_matrix) {
    int *row_start = calloc(M + 1, sizeof(int));
    if (!row_start) {
        fprintf(stderr, "Errore: allocazione memoria fallita per row_start.\n");
        return;
    }

    // Conta il numero di elementi non-zero per ogni riga
    for (int i = 0; i < nz; i++) {
        row_start[row_indices[i] + 1]++;
    }

    // Cumulativo per ottenere l'inizio di ogni riga
    for (int i = 1; i <= M; i++) {
        row_start[i] += row_start[i - 1];
    }

    // Parallelizzazione
    //#pragma omp parallel for
    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int start_row = block_idx * HackSize;
        int end_row = (block_idx + 1) * HackSize;
        if (end_row > M) end_row = M;
        int rows_in_block = end_row - start_row;

        // Calcolare il massimo numero di non-zero per riga nel blocco
        calculate_max_nz_per_row_in_block(M, row_indices, nz, block_idx, start_row, end_row, hll_matrix);

        int max_nz_per_row = hll_matrix->blocks[block_idx].max_nz_per_row;
        int size_of_arrays = max_nz_per_row * rows_in_block;
        printf("Rows in block %d: ", rows_in_block);
        printf("Max nz per row: %d\n", max_nz_per_row);
        printf("Size of arrays: %d\n", size_of_arrays);

        // Allocazione memoria per i dati del blocco
        hll_matrix->blocks[block_idx].JA = malloc(size_of_arrays * sizeof(int));
        hll_matrix->blocks[block_idx].AS = malloc(size_of_arrays * sizeof(double));
        if (!hll_matrix->blocks[block_idx].JA || !hll_matrix->blocks[block_idx].AS) {
            fprintf(stderr, "Errore: Allocazione fallita per il blocco %d.\n", block_idx);
            continue;
        }

        // Popolamento delle strutture HLL per ogni riga nel blocco
        //#pragma omp parallel for
        for (int i = start_row; i < end_row; i++) {
            int row_offset = (i - start_row) * max_nz_per_row;
            int row_nz_start = row_start[i];
            int row_nz_end = row_start[i + 1];
            int pos = 0;
            int last_col_idx = -1;

            // Assegna i valori nel formato HLL
            for (int j = row_nz_start; j < row_nz_end; j++) {
                if (pos >= max_nz_per_row) break;
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = col_indices[j];
                hll_matrix->blocks[block_idx].AS[index] = values[j];
                last_col_idx = col_indices[j];
                pos++;
            }

            // Riempi con valori di default
            while (pos < max_nz_per_row) {
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = last_col_idx;
                hll_matrix->blocks[block_idx].AS[index] = 0;
                pos++;
            }
        }
        printf("Blocco %d completato\n", block_idx + 1);
    }

    printf("Conversione HLL terminata!\n");
    free(row_start);
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

    //debug per verificare che hll funzionasse

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
    printf("Controllo completato, tutti i valori di y sono corretti.\n");
}
