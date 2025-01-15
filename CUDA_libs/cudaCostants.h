#ifndef CUDACOSTANTS_H
#define CUDACOSTANTS_H

// Iterazioni per ogni matrice per la raccolta dei risultati e performance
#define ITERATION_PER_MATRIX 20

// Definizione di HackSize
#define HackSize 32  // Dimensione del blocco
#define XBD 512
#define YBD 2

const dim3 BLOCK_DIM(XBD, YBD);

#endif //CUDACOSTANTS_H
