#ifndef HLLTOOL_H
#define HLLTOOL_H

#include "data_structure.h" // Include la definizione di HLL_Matrix

void convert_to_hll_cuda(matrixData *matrix_data, HLL_Matrix *hll_matrix,HLL_Matrix *d_hll_matrix);

int find_max_nz_per_block(const int *nz_per_row, int start_row, int end_row) ;
int find_max_nz(const int *nz_per_row, int start_row, int end_row);
void calculate_max_nz_in_row_in_block(const struct matrixData *matrix_data, int *nz_per_row);
__global__ void matvec_Hll_cuda( const HLL_Matrix *hll_matrix,  const double *x,  double *y,  int max_row_in_matrix);

#endif //HLLTOOL_H
