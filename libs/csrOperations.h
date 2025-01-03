#ifndef CSRSERIALIZED_H
#define CSRSERIALIZED_H
#include "../libs/data_structure.h"

// Calcolo csr serializzato
struct matrixPerformance serial_csr(struct matrixData *matrix_data, double *x);

// Calcolo csr parallelo con OpenMP
struct matrixPerformance parallel_csr(struct matrixData *matrix_data, double *x);

#endif
