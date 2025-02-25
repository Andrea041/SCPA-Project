#ifndef CSRTOOL_H
#define CSRTOOL_H

/* Funzione per convertire la matrice */
void convert_to_csr(int M, int nz, const int *row_indices, const int *col_indices, const double *values, int **IRP, int **JA, double **AS);

/* Prodotto matrice-vettore serializzato */
__global__ void gpuMatVec_csr(const int *d_IRP, const int *d_JA, const double *d_AS, const double *d_x, double *d_y,int M) ;

__global__ void gpuMatVec_csr_sm_par(const int *d_IRP, const int *d_JA, const double *d_AS, const double *d_x, double *d_y, int M);

__global__ void gpuMatVec_csr_sm_seq(const int *d_IRP, const int *d_JA, const double *d_AS, const double *d_x, double *d_y, int M);

void matvec_csr(int M, const int *IRP, const int *JA, const double *AS, double *x, double *y);

#endif //CSRTOOL_H
