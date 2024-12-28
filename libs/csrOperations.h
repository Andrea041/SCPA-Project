#ifndef CSRSERIALIZED_H
#define CSRSERIALIZED_H

// Calcolo csr serializzato
struct matrixPerformance serial_csr(int M, int nz, int *row_indices, int *col_indices, double *values, double *x);

// Calcolo csr parallelo con OpenMP
struct matrixPerformance parallel_csr(const int M, const int nz, int *row_indices, int *col_indices, double *values, double *x);

#endif
