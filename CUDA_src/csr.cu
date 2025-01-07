#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "../libs/matrixLists.h"
#include "../libs/data_structure.h"
#include "../libs/costants.h"
#include "../libs/csrOperations.h"
#include "../libs/hll_Operations.h"


#ifdef USER_PIERFRANCESCO
#include "cjson/cJSON.h"
const char *base_path = "/home/pierfrancesco/Desktop/matrix/";
#elif defined(USER_ANDREA)
#include "cjson/cJSON.h"
const char *base_path = "/Users/andreaandreoli/matrix/";
#else
#include "../../cJSON/include/cjson/cJSON.h"
const char *base_path = "../../matrix/";
#endif

int main() {
    ensure_directory_exists("../result");
    ensure_directory_exists("../result/iteration");
    ensure_directory_exists("../result/final");

    const int num_matrices = sizeof(matrix_names) / sizeof(matrix_names[0]);

    // Creazione degli array JSON per questa matrice
    cJSON *parallel_array_csr_CUDA = cJSON_CreateArray();
    cJSON *parallel_array_hll_CUDA = cJSON_CreateArray();

    for (int i = 0; i < num_matrices; i++) {
        struct matrixData *matrix_data = malloc(sizeof(struct matrixData));
        if (!matrix_data) {
            perror("Errore nell'allocazione della memoria per matrix_data.");
            return EXIT_FAILURE;
        }

        printf("Calcolo su matrice: %s\n", matrix_names[i]);
        preprocess_matrix(matrix_data, i);

        double *x = malloc(matrix_data->N * sizeof(double));
        if (!x) {
            perror("Errore nell'allocazione della memoria per il vettore x.");
            free(matrix_data);
            return EXIT_FAILURE;
        }
        for (int j = 0; j < matrix_data->N; j++) {
            x[j] = 1.0;
        }

        for (int j = 0; j < ITERATION_PER_MATRIX; j++) {
            /*parallelo CUDA && CSR*/
            add_performance_to_array(matrix_names[i], matrix_data, x, parallel_array_csr_CUDA, parallel_csr);

            /*parallelo CUDA && HLL*/
            add_performance_to_array(matrix_names[i], matrix_data, x, parallel_array_hll_CUDA, parallel_hll);
        }

        free(x);
        free(matrix_data);
    }

    write_json_to_file("../result/iteration/par_CUDA_CSR.json", parallel_array_csr_CUDA);
    write_json_to_file("../result/iteration/par_CUDA_HLL.json", parallel_array_hll_CUDA);

    cJSON_Delete(parallel_array_csr_CUDA);
    cJSON_Delete(parallel_array_hll_CUDA);

    calculatePerformance("../result/iteration/par_CUDA_CSR.json", "../result/final/par_CUDA_CSR.json");
    calculatePerformance("../result/iteration/par_CUDA_HLL.json", "../result/final/par_CUDA_HLL.json");

    return EXIT_SUCCESS;
}