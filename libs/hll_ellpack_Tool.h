#ifndef HLLTOOL_H
#define HLLTOOL_H

// Funzione per convertire la matrice in formato HLL
void convert_to_ellpack(int M, int nz, const int *row_indices, const int *col_indices, const double *values, int **JA, double **AS, int *MAXNZ);

// Prodotto matrice-vettore utilizzando formato ELLPACK
void matvec_Hll(HLL_Matrix *hll_matrix, double *x, double *y);

#endif