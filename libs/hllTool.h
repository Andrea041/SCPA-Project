#ifndef HLLTOOL_H
#define HLLTOOL_H

// Funzione per convertire la matrice in formato HLL
void convert_to_ellpack(int M, int N, int nz, int *row_indices, int *col_indices, double *values, int **JA, double **AS, int *MAXNZ);

// Prodotto matrice-vettore utilizzando formato ELLPACK
void matvec_ellpack(int M, int MAXNZ, int *JA, double *AS, double *x, double *y);

#endif