#ifndef HLLTOOL_H
#define HLLTOOL_H

#include "data_structure.h" // Include la definizione di HLL_Matrix

void convert_to_hll_cuda(matrixData *matrix_data, HLL_Matrix *hll_matrix);

__global__ void matvec_Hll_cuda( const HLL_Matrix *hll_matrix,  const double *x,  double *y,  int max_row_in_matrix);

#endif //HLLTOOL_H
