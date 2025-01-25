#ifndef CSROPERATIONS_H
#define CSROPERATIONS_H

#include "../libs/data_structure.h"

matrixPerformance serial_csr_cuda(matrixData *matrix_data_host, double *x_h);

matrixPerformance parallel_csr_cuda_v1(matrixData *matrix_data_host, double *x_h);

matrixPerformance parallel_csr_cuda_v2(matrixData *matrix_data_host, double *x_h);

matrixPerformance parallel_csr_cuda_v3(matrixData *matrix_data_host, double *x_h);

double checkDifferencesCUDA(double *y_h, int matrix_row);

#endif //CSROPERATIONS_H
