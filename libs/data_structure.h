//
// Created by pierfrancesco on 12/24/24.
//
#ifndef DATA_STRUCTURE_H
#define DATA_STRUCTURE_H

#include <stdlib.h>

// Struttura per i dati della matrice
struct matrixData {
    int *row_indices;
    int *col_indices;
    double *values;
    int M;
    int N;
    int nz;
};

// Struttura per le performance
struct matrixPerformance {
    double seconds;
    double flops;
    double megaFlops;
};

// Struttura per i risultati seriali
struct matrixResultSerial {
    char nameMatrix[50];
    double seconds;
    struct matrixResultSerial *next;
};

struct matrixResultSerialFINAL {
    char nameMatrix[50];  // Nome della matrice
    double avarangeFlops; // Flops medi
    double avarangeMegaFlops; // MFLOPS medi
    double avarangeSeconds; // Tempo medio
    struct matrixResultSerialFINAL *nextNode; // Puntatore al prossimo nodo
};

#endif // DATA_STRUCTURE_H
