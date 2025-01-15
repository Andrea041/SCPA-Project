#ifndef HLLTOOL_H
#define HLLTOOL_H

#include "data_structure.h" // Include la definizione di HLL_Matrix

void convert_to_hll(matrixData *matrix_data, HLL_Matrix *hll_matrix);

__global__  void gpuMatVec_Hll(const HLL_Matrix *hll_matrix, const double *x, double *y/*, int num_threads, const int *start_block, const int *end_block,*/ ,int max_row_in_matrix);

#endif //HLLTOOL_H
