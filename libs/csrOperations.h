#ifndef CSRSERIALIZED_H
#define CSRSERIALIZED_H
#include "../libs/data_structure.h"

/* Calcolo serializzato */
struct matrixPerformance serial_csr(struct matrixData *matrix_data, double *x, int num_threads);

double checkDifferencesOpenMP(double *y_h, int matrix_row);

/* Calcolo parallelo con OpenMP */
struct matrixPerformance parallel_csr(struct matrixData *matrix_data, double *x, int num_threads);

#endif
