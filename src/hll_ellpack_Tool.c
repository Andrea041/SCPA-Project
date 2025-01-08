#include <stdlib.h>
#include "../libs/hll_Operations.h"
#include "../libs/data_structure.h"
#include "../libs/hll_ellpack_Tool.h"
#include <stdio.h>
#include <omp.h>
#include <string.h>

#include "../libs/costants.h"

// Funzione per convertire una matrice sparsa al formato HLL
void convert_to_hll(int M, int N, int nz, const int *row_indices, const int *col_indices, const double *values, HLL_Matrix *hll_matrix) {
    int max_nz_per_row = hll_matrix->max_nz_per_row;

    int *row_start = calloc(M, sizeof(int));
    if (!row_start) {
        fprintf(stderr, "Errore: allocazione memoria fallita per row_start.\n");
        exit(EXIT_FAILURE);
    }

    // Conta il numero di elementi non-zero per ogni riga
    for (int i = 0; i < nz; i++) {
        row_start[row_indices[i] + 1]++;
    }

    // Cumulativo per ottenere l'inizio di ogni riga
    for (int i = 1; i <= M; i++) {
        row_start[i] += row_start[i - 1];
    }

    /* Parallelizzazione dell'elaborazione dei blocchi tra i thread */
    //#pragma omp parallel for
    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        /* Identificazione della riga di inizio e fine di ciascun blocco */
        int start_row = block_idx * HackSize;
        int end_row = (block_idx + 1) * HackSize;
        if (end_row > M) end_row = M;
        int rows_in_block = end_row - start_row;

        /* Calcolo della dimensione della matrice che contiene gli elementi di JA e AS */
        int size_of_arrays = max_nz_per_row * rows_in_block;

        /* Allocazione della memoria necessaria */
        hll_matrix->blocks[block_idx].JA = (int *)malloc(size_of_arrays * sizeof(int));
        hll_matrix->blocks[block_idx].AS = (double *)malloc(size_of_arrays * sizeof(double));
        if (!hll_matrix->blocks[block_idx].JA || !hll_matrix->blocks[block_idx].AS) {
            fprintf(stderr, "Errore: Allocazione fallita per il blocco %d.\n", block_idx);
            exit(EXIT_FAILURE);
        }

        /* Inizializzazione dei blocchi */
        memset(hll_matrix->blocks[block_idx].JA, 0, size_of_arrays * sizeof(int));
        memset(hll_matrix->blocks[block_idx].AS, 0, size_of_arrays * sizeof(double));
        printf("A\n");
        /* Elaborazione di ciascuna riga del blocco preso in carica */
        for (int i = start_row; i < end_row; i++) {
            int row_offset = (i - start_row) * max_nz_per_row;

            /* Questo perché li abbiamo oranizzati come se fossero in una rappresentazione csr */
            int row_nz_start = row_start[i];
            int row_nz_end = row_start[i + 1];

            /* Azzeramento delle variabili */
            int pos = 0;
            int last_col_idx = -1;
            /* Gestione degli elementi non-zero per la riga */
            for (int j = row_nz_start; j < row_nz_end; j++) {
                if (pos >= max_nz_per_row) {
                    fprintf(stderr, "Errore: Troppi elementi nella riga %d.\n", i);
                    exit(EXIT_FAILURE);
                }

                /* Assegnazione dei valori in ciascun array del blocco */
                int index = row_offset + pos;
                /* Entrambi gli array sono indicizzabili con j in quanto la memorizzazione interna è 1 a 1 */
                hll_matrix->blocks[block_idx].JA[index] = col_indices[j];
                hll_matrix->blocks[block_idx].AS[index] = values[j];
                /* Tengo traccia del valore dell'ultimo indice di colonna per un eventuale padding */
                last_col_idx = col_indices[j];
                pos++;
            }
            /* Padding per righe con meno di max_nz_per_row */
            while (pos < max_nz_per_row) {
                /* Calcolo l'offset a cui ero rimasto precedentemente */
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = last_col_idx;
                hll_matrix->blocks[block_idx].AS[index] = 0.0;
                pos++;
            }
        }
        printf("Blocco %d completato\n", block_idx + 1);
    }

    free(row_start);
    //printf("Conversione HLL completata!\n");
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
