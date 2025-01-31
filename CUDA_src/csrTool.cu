#include <cstdlib>
#include <cstdio>
#include <helper_cuda.h>

#include "../CUDA_libs/csrTool.h"

/* Funzione per convertire la matrice in formato CSR */
void convert_to_csr(int M, int nz, const int *row_indices, const int *col_indices, const double *values, int **IRP, int **JA, double **AS) {
    *IRP = static_cast<int *>(malloc((M + 1) * sizeof(int)));    // Dimensione del vettore M
    *JA = static_cast<int *>(malloc(nz * sizeof(int)));          // Dimensione del vettore NZ - 1
    *AS = static_cast<double *>(malloc(nz * sizeof(double)));    // Dimensione del vettore NZ - 1

    if (*IRP == nullptr || *JA == nullptr || *AS == nullptr) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        exit(1);
    }

    for (int i = 0; i <= M; i++) {
        (*IRP)[i] = 0;
    }

    for (int i = 0; i < nz; i++) {
        if (row_indices[i] < 0 || row_indices[i] >= M) {
            fprintf(stderr, "Errore: l'indice di riga è fuori dai limiti.\n");
            exit(1);
        }
        (*IRP)[row_indices[i] + 1]++;
    }

    for (int i = 0; i < M; i++) {
        (*IRP)[i + 1] += (*IRP)[i];
    }

    int *row_position = static_cast<int *>(malloc(M * sizeof(int)));
    if (row_position == nullptr) {
        fprintf(stderr, "Errore nell'allocazione di row_position.\n");
        exit(1);
    }

    for (int i = 0; i < M; i++) {
        row_position[i] = 0;
    }

    for (int i = 0; i < nz; i++) {
        int row = row_indices[i];
        int pos = (*IRP)[row] + row_position[row];
        (*JA)[pos] = col_indices[i];
        (*AS)[pos] = values[i];
        row_position[row]++;
    }

    free(row_position);
}

/* Prodotto matrice-vettore serializzato su CPU */
void matvec_csr(int M, const int *IRP, const int *JA, const double *AS, double *x, double *y) {
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
        for (int j = IRP[i]; j < IRP[i + 1]; j++) {
            y[i] += AS[j] * x[JA[j]];
        }
    }
}

/* Prodotto matrice-vettore parallelo su GPU - ciascun thread di un blocco prende in carico una riga */
__global__ void gpuMatVec_csr(const int *d_IRP, const int *d_JA, const double *d_AS, const double *d_x, double *d_y, int M) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    /* Controllo che la riga non ecceda il numero totale di righe */
    if (row >= M) return;

    double sum = 0.0;
    /* Ciascun thread si fa carico di una riga */
    for (int index = d_IRP[row]; index < d_IRP[row + 1]; index++)
        sum += d_AS[index] * d_x[d_JA[index]];
    d_y[row] = sum;
}

/* Prodotto matrice-vettore parallelo su GPU - shared memory e riduzione sequenziale */
__global__ void gpuMatVec_csr_sm_seq(const int *d_IRP, const int *d_JA, const double *d_AS, const double *d_x, double *d_y, int M) {
    const int bidx = blockIdx.x;  // Indice del blocco
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    extern __shared__ double sharedMem[];
    sharedMem[tid] = 0.0;

    if (bidx >= M) return;

    const int start = d_IRP[bidx];
    const int end = d_IRP[bidx + 1];

    if (tid >= end - start) return;

    double sum = 0.0;
    for (int i = tid; i < end - start; i += blockDim.x) {
        sum += d_AS[start + i] * d_x[d_JA[start + i]];
    }
    sharedMem[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            sharedMem[tid] += sharedMem[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        d_y[bidx] += sharedMem[0];
}

__device__ void warpReduce(volatile double *sdata, const int tid) {
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

/* Prodotto matrice-vettore parallelo su GPU - shared memory e riduzione parallela (ciascun blocco ha una riga e ciascun thread una componente) */
__global__ void gpuMatVec_csr_sm_par(const int *d_IRP, const int *d_JA, const double *d_AS, const double *d_x, double *d_y, int M) {
    const int bidx = blockIdx.x;  // Indice del blocco (riga della matrice)
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    /* Inizializzazione shared memory */
    extern __shared__ double sharedMem[];
    sharedMem[tid] = 0.0;

    /* Controllo per vedere se l'indice di blocco supera la dimensione del vettore IRP */
    if (bidx >= M) return;

    const int start = d_IRP[bidx];
    const int end = d_IRP[bidx + 1];

    /* Evitiamo che i thread vadano a scrivere in zone di memoria condivise "non concesse" */
    if (tid >= end - start) return;

    /* Prodotto parallelo */
    double sum = 0.0;
    for (int i = tid; i < end - start; i += blockDim.x) {
        sum += d_AS[start + i] * d_x[d_JA[start + i]];
    }
    sharedMem[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s) {
            sharedMem[tid] += sharedMem[tid + s];
        }
        __syncthreads();
    }

    /* Riduzione del primo warp */
    if (tid < 32)
        warpReduce(sharedMem, tid);

    /* Scrittura (da parte del tid 0) del risultato dato dalla riduzione che si troverà nella entry 0-esima su memoria globale */
    if (tid == 0)
        d_y[bidx] += sharedMem[0];
}