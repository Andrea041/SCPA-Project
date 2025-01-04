#ifndef CSR_H
#define CSR_H

// Funzione per convertire la matrice
void convert_to_csr(int M, int nz, const int *row_indices, const int *col_indices, const double *values, int **IRP, int **JA, double **AS);

// Prodotto matrice-vettore serializzato
void matvec_csr(int M, const int *IRP, const int *JA, const double *AS, double *x, double *y);

// Prodotto matrice-vettore parallelizzato con OpenMP
void matvec_csr_openMP(const int *IRP, const int *JA, const double *AS, const double *x, double *y,int *start_row, int *end_row, int num_threads, int nz, int M);

#endif
