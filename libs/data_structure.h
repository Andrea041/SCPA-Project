// Created by pierfrancesco on 12/24/24.
//

#ifndef DATA_STRUCTURE_H
#define DATA_STRUCTURE_H

#include "mmio.h"
#include <stdint.h>

// Struttura per i dati della matrice
struct matrixData {
    int *row_indices;
    int *col_indices;
    double *values;
    int M;
    int N;
    int nz;
    MM_typecode matcode;
};

// Struttura per memorizzare i dati di un singolo blocco ELLPACK
typedef struct {
    int *JA;   // Indici delle colonne
    double *AS;
    int size_of_arrays;
    int max_nz_per_row;     // Numero massimo di non nulli per riga
    int nz_per_block;
} ELLPACK_Block;

// Struttura per memorizzare l'intera matrice HLL
typedef struct {
    ELLPACK_Block *blocks;  // Array di blocchi ELLPACK
    int num_blocks;         // Numero di blocchi
} HLL_Matrix;

// Struttura per le performance
struct matrixPerformance {
    char nameMatrix[50];
    double seconds;
    double flops;
    double gigaFlops;
};

struct matrixPerformanceAverage {
    char nameMatrix[50];  // Nome della matrice
    double avarangeFlops; // Flops medi
    double avarangeMegaFlops; // MFLOPS medi
    double avarangeSeconds; // Tempo medio
};

typedef struct {
    char nameMatrix[50];
    double total_seconds;
    int count;
    int nz;
    int row;
    int col;
} MatrixPerformanceResult;

#endif // DATA_STRUCTURE_H
