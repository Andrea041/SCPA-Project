#ifndef CUDACOSTANTS_H
#define CUDACOSTANTS_H

/* Dimensione del blocco su ciascuna coordinata -> blocchi 2D di dimensione (BDX x BDY) -> Si possono avere massimo 1024 */
#define BDX 512
#define BDY 2
#define WARP_SIZE 32

constexpr dim3 BLOCK_DIM(BDX, BDY);

/* Dimensione della shared memory -> limita il parallelismo a SHARED_MEM_SIZE threads */
#define SHARED_MEM_SIZE 1024 * sizeof(double)
#define SHARED_MEM_SIZE_HLL 192

#endif //CUDACOSTANTS_H
