#ifndef CSR_H
#define CSR_H

// Funzione per convertire la matrice
void convert_to_csr(int M, int nz, const int *row_indices, const int *col_indices, double *values, int **IRP, int **JA, double **AS);

// Prodotto matrice-vettore
void matvec_csr(int M, const int *IRP, int *JA, double *AS, double *x, double *y);

#endif
