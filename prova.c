#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <bits/time.h>

#include "mmio.h"
const char *matrix_names[] = {
    "cage4.mtx",
   /*"mhda416.mtx",
   "mcfe.mtx",
   "olm1000.mtx",
   "adder_dcop_32.mtx",
   "west2021.mtx",
   "cavity10.mtx",
   "rdist2.mtx",*/
   /*"cant.mtx",
   /*"olafu.mtx",
   "Cube_Coup_dt0.mtx",
   "ML_Laplace.mtx",
   "bcsstk17.mtx",
   "mac_econ_fwd500.mtx",
   "mhd4800a.mtx",
   "cop20k_A.mtx",
   "raefsky2.mtx",
   "af23560.mtx",
   "lung2.mtx",
   "PR02R.mtx",
   "FEM_3D_thermal1.mtx",
   "thermal1.mtx",
   "thermal2.mtx",
   "thermomech_TK.mtx",
   "nlpkkt80.mtx",
   "webbase-1M.mtx",
   "dc1.mtx",
   "amazon0302.mtx",
   "af_1_k101.mtx",
   "roadNet-PA.mtx"*/
};

const char *base_path = "/home/pierfrancesco/Desktop/SCPA-Project/matrix/";

// Funzione per convertire la matrice in formato HLL
void convert_to_ellpack(int M, int N, int nz, int *row_indices, int *col_indices, double *values, int **JA, double **AS, int *MAXNZ) {
    *MAXNZ = 0;

    // Trova il massimo numero di non-zeri per riga
    int *row_counts = (int *)calloc(M, sizeof(int));
    for (int i = 0; i < nz; i++) {
        row_counts[row_indices[i]]++;
    }
    for (int i = 0; i < M; i++) {
        if (row_counts[i] > *MAXNZ) {
            *MAXNZ = row_counts[i];
        }
    }

    // Alloca memoria per JA e AS
    *JA = (int *)malloc(M * (*MAXNZ) * sizeof(int));
    *AS = (double *)malloc(M * (*MAXNZ) * sizeof(double));

    // Inizializza JA e AS con valori di default
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < *MAXNZ; j++) {
            (*JA)[i * (*MAXNZ) + j] = (j == 0) ? 1 : (*JA)[i * (*MAXNZ) + j - 1]; // Indice valido o precedente
            (*AS)[i * (*MAXNZ) + j] = 0.0; // Valore zero per celle vuote
        }
    }

    // Popola JA e AS
    int *row_position = (int *)calloc(M, sizeof(int));
    for (int i = 0; i < nz; i++) {
        int row = row_indices[i];
        int pos = row_position[row];
        (*JA)[row * (*MAXNZ) + pos] = col_indices[i] + 1; // Indici 1-based
        (*AS)[row * (*MAXNZ) + pos] = values[i];
        row_position[row]++;
    }

    free(row_counts);
    free(row_position);
}

// Funzione per convertire la matrice in formato CSR
void convert_to_csr(int M, int N, int nz, int *row_indices, int *col_indices, double *values, int **IRP, int **JA, double **AS) {
    *IRP = (int *)malloc((M + 1) * sizeof(int)); // Array dei puntatori di riga
    *JA = (int *)malloc(nz * sizeof(int));       // Array degli indici di colonna
    *AS = (double *)malloc(nz * sizeof(double)); // Array dei valori non nulli

    // Inizializza IRP a zero
    for (int i = 0; i <= M; i++) {
        (*IRP)[i] = 0;
    }

    // Conta i non-zero per ogni riga
    for (int i = 0; i < nz; i++) {
        (*IRP)[row_indices[i] + 1]++;
    }

    // Costruisci IRP come somma cumulativa
    for (int i = 0; i < M; i++) {
        (*IRP)[i + 1] += (*IRP)[i];
    }

    // Popola JA e AS
    int *row_position = (int *)calloc(M, sizeof(int));
    for (int i = 0; i < nz; i++) {
        int row = row_indices[i];
        int pos = (*IRP)[row] + row_position[row];
        (*JA)[pos] = col_indices[i];
        (*AS)[pos] = values[i];
        row_position[row]++;
    }

    free(row_position);
}

// Prodotto matrice-vettore utilizzando formato CSR
void matvec_csr(int M, int *IRP, int *JA, double *AS, double *x, double *y) {
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
        for (int j = IRP[i]; j < IRP[i + 1]; j++) {
            y[i] += AS[j] * x[JA[j]];
        }
    }
}

// Prodotto matrice-vettore utilizzando formato ELLPACK
void matvec_ellpack(int M, int MAXNZ, int *JA, double *AS, double *x, double *y) {
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
        for (int j = 0; j < MAXNZ; j++) {
            int col = JA[i * MAXNZ + j];
            if (col != -1) { // Controlla se l'indice di colonna è valido
                y[i] += AS[i * MAXNZ + j] * x[col];
            }
        }
    }
}

int main() {
    const int num_matrices = sizeof(matrix_names) / sizeof(matrix_names[0]); // Numero di matrici
    for (int i = 0; i < num_matrices; i++) {
        char full_path[512]; // Percorso completo
        snprintf(full_path, sizeof(full_path), "%s%s", base_path, matrix_names[i]);
        printf("Percorso completo: %s\n", full_path);

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

        int *row_indices = (int *)malloc(nz * sizeof(int));
        int *col_indices = (int *)malloc(nz * sizeof(int));
        double *values = (double *)malloc(nz * sizeof(double));

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

       int *IRP, *JA; //Formato CSR
        double *AS;
        convert_to_csr(M, N, nz, row_indices, col_indices, values, &IRP, &JA, &AS);

       /* int *IRP, *JA;           // Indici di colonna per il formato ELLPACK
        double *AS;        // Coefficienti della matrice in ELLPACK
        int MAXNZ;         // Numero massimo di non-zeri per riga
        convert_to_ellpack(M, N, nz, row_indices, col_indices, values, &JA, &AS, &MAXNZ);*/


        double *x = (double *)malloc(N * sizeof(double));
        double *y = (double *)malloc(M * sizeof(double));

        // Inizializza il vettore x con tutti 1
        for (int j = 0; j < N; j++) {
            x[j] = 1.0;
        }

        // Misura il tempo per il prodotto matrice-vettore
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        matvec_csr(M, IRP, JA, AS, x, y); //per formsto CSR
       // matvec_ellpack(M, MAXNZ, JA, AS, x, y); //per formsto HLL

        clock_gettime(CLOCK_MONOTONIC, &end);

        double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        // Calcolo delle performance: FLOPS e MFLOPS
        double flops = 2.0 * nz / time_spent;
        double mflops = flops / 1e6;

        // (Opzionale) Stampa il risultato
        printf("Risultato del prodotto matrice-vettore:\n");
        for (int j = 0; j < M; j++) {
            printf("y[%d] = %.2f\n", j, y[j]);
        }

        printf("Tempo per il prodotto matrice-vettore: %.6f secondi\n", time_spent);
        printf("Performance: %.2f FLOPS\n", flops);
        printf("Performance: %.2f MFLOPS\n", mflops);


    }

    return EXIT_SUCCESS;
}


int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz )
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;

    /* set return null parameter values, in case we exit with errors */
    *M = *N = *nz = 0;

    /* now continue scanning until you reach the end-of-comments */
    do
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL)
            return MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d %d", M, N, nz) == 3)
        return 0;

    else
        do
        {
            num_items_read = fscanf(f, "%d %d %d", M, N, nz);
            if (num_items_read == EOF) return MM_PREMATURE_EOF;
        }
    while (num_items_read != 3);

    return 0;
}

int mm_read_banner(FILE *f, MM_typecode *matcode)
{
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH];
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;


    mm_clear_typecode(matcode);

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
        return MM_PREMATURE_EOF;

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type,
        storage_scheme) != 5)
        return MM_PREMATURE_EOF;

    for (p=mtx; *p!='\0'; *p=tolower(*p),p++);  /* convert to lower case */
    for (p=crd; *p!='\0'; *p=tolower(*p),p++);
    for (p=data_type; *p!='\0'; *p=tolower(*p),p++);
    for (p=storage_scheme; *p!='\0'; *p=tolower(*p),p++);

    /* check for banner */
    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
        return MM_NO_HEADER;

    /* first field should be "mtx" */
    if (strcmp(mtx, MM_MTX_STR) != 0)
        return  MM_UNSUPPORTED_TYPE;
    mm_set_matrix(matcode);


    /* second field describes whether this is a sparse matrix (in coordinate
            storgae) or a dense array */


    if (strcmp(crd, MM_SPARSE_STR) == 0)
        mm_set_sparse(matcode);
    else
    if (strcmp(crd, MM_DENSE_STR) == 0)
            mm_set_dense(matcode);
    else
        return MM_UNSUPPORTED_TYPE;


    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(matcode);
    else
    if (strcmp(data_type, MM_COMPLEX_STR) == 0)
        mm_set_complex(matcode);
    else
    if (strcmp(data_type, MM_PATTERN_STR) == 0)
        mm_set_pattern(matcode);
    else
    if (strcmp(data_type, MM_INT_STR) == 0)
        mm_set_integer(matcode);
    else
        return MM_UNSUPPORTED_TYPE;


    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(matcode);
    else
    if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
        mm_set_symmetric(matcode);
    else
    if (strcmp(storage_scheme, MM_HERM_STR) == 0)
        mm_set_hermitian(matcode);
    else
    if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
        mm_set_skew(matcode);
    else
        return MM_UNSUPPORTED_TYPE;


    return 0;
}