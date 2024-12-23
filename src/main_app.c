#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../libs/csrSerialized.h"
#include "../libs/mmio.h"
#include "../libs/matrixLists.h"

#ifdef USER_PIERFRANCESCO
const char *base_path = "/home/pierfrancesco/Desktop/SCPA-Project/matrix/";
#elif defined(USER_ANDREA)
const char *base_path = "/Users/andreaandreoli/matrix/";
#else
const char *base_path = "./matrix/"; // Valore predefinito
#endif

struct matrixData {
    int *row_indices;
    int *col_indices;
    double *values;
};

struct matrixConst {
    int M;
    int N;
    int nz;
};

/* Funzione per il preprocessamento delle matrici in input da file */
struct matrixConst preprocess_matrix(struct matrixData *matrix_data, int i) {
    char full_path[512];
    struct matrixConst matrixConst;
    snprintf(full_path, sizeof(full_path), "%s%s", base_path, matrix_names[i]);

    FILE *f = fopen(full_path, "r");
    if (!f) {
        perror("Errore nell'apertura del file");
        exit(EXIT_FAILURE);
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "Errore nella lettura del banner Matrix Market.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode)) {
        fprintf(stderr, "Il file non è in formato matrice sparsa a coordinate.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    int M, N, nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        fprintf(stderr, "Errore nella lettura delle dimensioni della matrice.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    matrixConst.M = M;
    matrixConst.N = N;
    matrixConst.nz = nz;

    matrix_data->row_indices = malloc(nz * sizeof(int));
    matrix_data->col_indices = malloc(nz * sizeof(int));
    matrix_data->values = malloc(nz * sizeof(double));

    if (matrix_data->row_indices == NULL || matrix_data->col_indices == NULL || matrix_data->values == NULL) {
        perror("Errore nell'allocazione della memoria.");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < nz; j++) {
        if (fscanf(f, "%d %d %lf", &matrix_data->row_indices[j], &matrix_data->col_indices[j], &matrix_data->values[j]) != 3) {
            fprintf(stderr, "Errore nella lettura degli elementi della matrice.\n");
            free(matrix_data->row_indices);
            free(matrix_data->col_indices);
            free(matrix_data->values);
            fclose(f);
            exit(EXIT_FAILURE);
        }
        matrix_data->row_indices[j]--; // Converti a indice 0-based
        matrix_data->col_indices[j]--; // Converti a indice 0-based
    }
    fclose(f);

    return matrixConst;
}

int main() {
    const int num_matrices = sizeof(matrix_names) / sizeof(matrix_names[0]); // Numero di matrici
    struct matrixData *matrix_data = malloc(sizeof(struct matrixData));
    if (matrix_data == NULL) {
        perror("Errore nell'allocazione della memoria per matrix_data.");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < num_matrices; i++) {
        struct matrixConst matrix_const = preprocess_matrix(matrix_data, i);

        // Inizializzazione del vettore di input x
        double *x = malloc((size_t)matrix_const.N * sizeof(double));
        if (x == NULL) {
            perror("Errore nell'allocazione della memoria per il vettore x.");
            free(matrix_data);
            return EXIT_FAILURE;
        }
        for (int j = 0; j < matrix_const.N; j++) {
            x[j] = 1.0;
        }

        serial_csr(matrix_const.M, matrix_const.N, matrix_const.nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, x);

        free(x);
        free(matrix_data->row_indices);
        free(matrix_data->col_indices);
        free(matrix_data->values);
        free(matrix_data);
    }

    return EXIT_SUCCESS;
}
