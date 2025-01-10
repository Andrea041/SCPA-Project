#ifndef HLL_ELLPACK_TOOL_H
#define HLL_ELLPACK_TOOL_H

#include "data_structure.h" // Include la definizione di HLL_Matrix

void convert_to_hll(struct matrixData *matrix_data, HLL_Matrix *hll_matrix);

void matvec_Hll(HLL_Matrix *hll_matrix, double *x, double *y, int num_threads, int *start_row, int *end_row, int N,int M) ;

#endif // HLL_ELLPACK_TOOL_H
