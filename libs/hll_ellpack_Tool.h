#ifndef HLL_ELLPACK_TOOL_H
#define HLL_ELLPACK_TOOL_H

#include "data_structure.h" // Include la definizione di HLL_Matrix

void convert_to_hll(struct matrixData *matrix_data, HLL_Matrix *hll_matrix);

void matvec_Hll(const HLL_Matrix *hll_matrix, const double *x, double *y, int num_threads, const int *start_block, const int *end_block, int max_row_in_matrix); ;

void calculate_max_nz_in_row_in_block(const struct matrixData *matrix_data, int *nz_per_row);

#endif // HLL_ELLPACK_TOOL_H
