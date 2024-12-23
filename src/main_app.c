/*
 * Questo è il punto di accesso principale al programma di calcolo
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../libs/csrSerialized.h"
#include "../libs/mmio.h"
#include "../libs/matrixLists.h"
#include "../libs/csrTool.h"

#ifdef USER_PIERFRANCESCO
const char *base_path = "/home/pierfrancesco/Desktop/SCPA-Project/matrix/";
#elif defined(USER_ANDREA)
const char *base_path = "/Users/andreaandreoli/matrix/";
#else
const char *base_path = "./matrix/"; // Valore predefinito
#endif

int main() {
    const int num_matrices = sizeof(matrix_names) / sizeof(matrix_names[0]); // Numero di matrici
    for (int i = 0; i < num_matrices; i++) {
        char full_path[512]; // Percorso completo
        snprintf(full_path, sizeof(full_path), "%s%s", base_path, matrix_names[i]);

        FILE *f = fopen(full_path, "r");
        if (!f) {
            perror("Errore nell'apertura del file");
            return EXIT_FAILURE;
        }

        MM_typecode matcode;
        if (mm_read_banner(f, &matcode) != 0) {
            fprintf(stderr, "Errore nella lettura del banner Matrix Market.\n");
            fclose(f);
            return EXIT_FAILURE;
        }

        if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode)) {
            fprintf(stderr, "Il file non è in formato matrice sparsa a coordinate.\n");
            fclose(f);
            return EXIT_FAILURE;
        }

        int M, N, nz;
        if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
            fprintf(stderr, "Errore nella lettura delle dimensioni della matrice.\n");
            fclose(f);
            return EXIT_FAILURE;
        }

        int *row_indices = malloc(nz * sizeof(int));
        int *col_indices = malloc(nz * sizeof(int));
        double *values = malloc(nz * sizeof(double));

        for (int j = 0; j < nz; j++) {
            if (fscanf(f, "%d %d %lf", &row_indices[j], &col_indices[j], &values[j]) != 3) {
                fprintf(stderr, "Errore nella lettura degli elementi della matrice.\n");
                free(row_indices);
                free(col_indices);
                free(values);
                fclose(f);
                return EXIT_FAILURE;
            }
            row_indices[j]--; // Converti a indice 0-based
            col_indices[j]--; // Converti a indice 0-based
        }
        fclose(f);

        /* Inizializzazione del vettore di input x */
        double *x = malloc((size_t)N * sizeof(double));
        for (int j = 0; j < N; j++) {
            x[j] = 1.0;
        }

        serial_csr(M, N, nz, row_indices, col_indices, values, x);

       /* int *IRP, *JA;           // Indici di colonna per il formato ELLPACK
        double *AS;        // Coefficienti della matrice in ELLPACK
        int MAXNZ;         // Numero massimo di non-zeri per riga
        convert_to_ellpack(M, N, nz, row_indices, col_indices, values, &JA, &AS, &MAXNZ);
        matvec_ellpack(M, MAXNZ, JA, AS, x, y); //per formsto HLL
        */

       free(x);
    }
    return EXIT_SUCCESS;
}