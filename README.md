# SCPA-Project

# Project: Parallel Computing Systems and Applications

## Description
This project implements a computation kernel for the multiplication of a sparse matrix and a vector:

y ← Ax

with the matrix A stored in the following formats:
- **CSR (Compressed Sparse Row)**
- **HLL (Hybrid of ELLPACK and List)**

The implementation leverages **OpenMP** and **CUDA** parallelization techniques to enhance computation performance.

## Technologies Used
- **Language**: C
- **Parallelization**: OpenMP, CUDA
- **Testing and Validation**: Comparison with a minimal serial implementation in CSR
- **Test Dataset**: Matrices from the [Suite Sparse Matrix Collection](https://sparse.tamu.edu/)

## Storage Formats
### CSR (Compressed Sparse Row)
Each M × N matrix with NZ non-zero elements is stored with:
- `IRP(1:M+1)`: Pointers to the beginning of each row
- `JA(1:NZ)`: Column indices
- `AS(1:NZ)`: Coefficient values

### HLL (Hybrid of ELLPACK and List)
- The matrix is divided into blocks of `HackSize` rows
- Each block is stored in **ELLPACK** format

### ELLPACK
- `JA(1:M,1:MAXNZ)`: Column indices
- `AS(1:M,1:MAXNZ)`: Coefficient value
