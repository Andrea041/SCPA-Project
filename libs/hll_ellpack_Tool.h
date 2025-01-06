#ifndef HLL_ELLPACK_TOOL_H
#define HLL_ELLPACK_TOOL_H

#include "data_structure.h" // Include la definizione di HLL_Matrix

void convert_to_hll(int M, int N, int nz, const int *row_indices, const int *col_indices, const double *values, HLL_Matrix *hll_matrix);
void matvec_Hll(HLL_Matrix *hll_matrix, double *x, double *y);

#endif // HLL_ELLPACK_TOOL_H
