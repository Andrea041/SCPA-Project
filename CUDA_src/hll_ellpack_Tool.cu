#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "../libs/data_structure.h"
#include "../libs/hll_ellpack_Tool.h"
#include "../libs/costants.h"
#include "../CUDA_libs/hllTool.h"

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
void convert_to_hll_cuda( matrixData *matrix_data, HLL_Matrix *hll_matrix, HLL_Matrix *d_hll_matrix) {
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
        hll_matrix->blocks[block_idx].size_of_arrays=size_of_arrays ;
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

/* Prodotto matrice-vettore parallelo su GPU - ciascun thread di un blocco prende in carico una riga */
__global__ void matvec_Hll_cuda(const HLL_Matrix *d_hll_matrix, const double *d_x, double *d_y, int M) {
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_row >= M) return;

    // Identifico il blocco a livello globale
    int block_id = global_row / HackSize;
    // Identifico la riga all'interno del blocco
    int local_row = global_row % HackSize;

    const ELLPACK_Block *block = &d_hll_matrix->blocks[block_id];

    // Individuo la riga da cui devo partire ad effettuare il prodotto
    int row_offset = local_row * block->max_nz_per_row;

    // Calcola il prodotto matrice-vettore
    double sum = 0.0;
    for (int j = 0; j < block->max_nz_per_row; j++)
        sum += block->AS[row_offset + j] * d_x[block->JA[row_offset + j]];

    d_y[global_row] = sum;
}

/*__device__ void process_other_blocks(ELLPACK_Block *block, const double *d_x, double *d_y, int M, const int global_row, const int global_col,
                                    int block_number, int tid, int block_x, volatile double *sharedMem, int local_index) {
    int aux_block = blockIdx.y;
    if (aux_block != block_number || blockIdx.x != block_x) return;

    const int global_index = global_row * block->max_nz_per_row + global_col;
    if (global_index >= block->max_nz_per_row * M) return;

    // Memoria condivisa per il blocco
    sharedMem[local_index] = 0.0;  // Inizializzazione a 0
    __syncthreads();

    // Accumulo del prodotto matrice-vettore
    double sum = 0.0;
    sum = block->AS[global_index] * d_x[block->JA[global_index]];
    sharedMem[local_index] = sum;

    // Riduzione parallela per ciascuna riga
    if (local_index % block->max_nz_per_row == 0) {
        for (int j = 1; j < block->max_nz_per_row; j++)
            sharedMem[local_index] += sharedMem[local_index + j];
    }
    __syncthreads();

    int row = global_index / block->max_nz_per_row;
    if (tid == 0) {
        for (int i = 0; i < block->max_nz_per_row * M; i += block->max_nz_per_row) {
            d_y[row] += sharedMem[i];
            row++;
        }
    }
}*/

/* Prodotto matrice-vettore parallelo su GPU - ciascun blocco prende in carico un blocco ellpack e ciascun thread si occupa di svolgere il prodotto per una componente */
__global__ void matvec_Hll_cuda_SH(const HLL_Matrix *d_hll_matrix, const double *d_x, double *d_y, int M, int N) {
    // Indici globali di riga e colonna (blocchi 2D)
    const int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_thread_idx = threadIdx.y * blockDim.x + threadIdx.x; // Indice locale nel blocco
    int block_x = blockIdx.x;

    // Controllo limiti globali
    if (global_row >= M || global_col >= N) return;

    // Identifico il blocco a livello globale
    int main_block = global_row / HackSize;
    // Identifico la riga all'interno del blocco
    int local_row = global_row % HackSize;
    if (main_block >= d_hll_matrix->num_blocks || block_x != main_block) return;

    ELLPACK_Block *block = &d_hll_matrix->blocks[main_block];

    // Decommentare se si riesce ad implementare la gestione nel caso in cui la matrice abbia un numero di nonzeri maggiore di 32
    /*int auxiliary_blocks = 0;
    if (block->max_nz_per_row >= 32) {
        // Colonne rimanenti dopo aver processato il primo blocco
        int remaining_columns = block->max_nz_per_row - 32;
        // Calcolo quanti blocchi mi servono per completare il calcolo di un singolo blocco JA - AS
        auxiliary_blocks = remaining_columns / 32 + 1;
    }*/

    int block_start_col = 0;
    for (int i = 0; i < main_block; i++) {
        block_start_col += d_hll_matrix->blocks[i].max_nz_per_row;
    }
    int local_col = global_col - block_start_col;


    const int local_index = local_row * block->max_nz_per_row + local_col;
    if (local_index >= block->max_nz_per_row * M) return;

    // Memoria condivisa per il blocco
    extern __shared__ double sharedMem[];
    sharedMem[local_index] = 0.0;  // Inizializzazione a 0

    // Decommentare se si riesce ad implementare la gestione nel caso in cui la matrice abbia un numero di nonzeri maggiore di 32
    /*if (auxiliary_blocks > 0) {
        for (int block_number = 1; block_number < auxiliary_blocks + 1; block_number++) {
            //process_other_blocks(block, d_x, d_y, M, global_row, global_col, block_number, local_thread_idx, main_block, sharedMem, local_index);
        }
    }*/

    // Accumulo del prodotto matrice-vettore
    double sum = 0.0;
    sum = block->AS[local_index] * d_x[block->JA[local_index]];
    sharedMem[local_index] = sum;

    // Riduzione parallela per ciascuna riga
    if (local_index % block->max_nz_per_row == 0) {
        for (int j = 1; j < block->max_nz_per_row; j++)
            sharedMem[local_index] += sharedMem[local_index + j];
    }
    __syncthreads();

    int row = 0;
    if (local_thread_idx == 0) {
        for (int i = 0; i < block->max_nz_per_row * M; i += block->max_nz_per_row) {
            d_y[row] += sharedMem[i];
            row++;
        }
    }
}

