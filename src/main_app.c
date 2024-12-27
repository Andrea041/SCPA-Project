#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "../libs/csrOperations.h"
#include "../libs/mmio.h"
#include "../libs/matrixLists.h"
#include "../libs/data_structure.h"

#ifdef USER_PIERFRANCESCO
const char *base_path = "/home/pierfrancesco/Desktop/matrix/";
#elif defined(USER_ANDREA)
const char *base_path = "/Users/andreaandreoli/matrix/";
#else
const char *base_path = "./matrix/";
#endif

#define ITERATION_PER_MATRIX 5

/* Funzione per stampare i risultati dalla lista */
void print_results(struct matrixResultSerialFINAL *results) {
    printf("Lista dei risultati:\n");
    struct matrixResultSerialFINAL *current = results;
    while (current != NULL) {
        printf("Matrix name: %s\n", current->nameMatrix);
        printf("Mean execution time: %f\n", current->avarangeSeconds);
        printf("FLOPS: %f\n", current->avarangeFlops);
        printf("MFLOPS: %f\n", current->avarangeMegaFlops);
        current = current->nextNode;
    }
}

/* Funzione per il reset della struttura dati utilizzata per la memorizzazione di una matrice */
void clean_matrix_mem(struct matrixData *matrix_data) {
    free(matrix_data->col_indices);
    free(matrix_data->row_indices);
    free(matrix_data->values);
    matrix_data->M = 0;
    matrix_data->N = 0;
    matrix_data->nz = 0;
}

/* Funzione per il calcolo delle prestazioni */
struct matrixResultSerialFINAL *calculate_performance(struct matrixResultSerial *head, const struct matrixData *matrix_data, const char *matrix_name) {
    struct matrixResultSerialFINAL *node = malloc(sizeof(struct matrixResultSerialFINAL));
    if (node == NULL) {
        perror("Errore nell'allocazione della memoria per matrixResultSerialFINAL");
        exit(EXIT_FAILURE); // Termina il programma in caso di errore
    }

    double sum = 0.0;
    int count = 0;
    struct matrixResultSerial *current = head;

    while (current != NULL) {
        sum += current->seconds;    // Somma i secondi
        count++;                    // Conta il numero di nodi
        current = current->next;    // Passa al nodo successivo
    }

    if (count == 0) {
        fprintf(stderr, "Errore: la lista è vuota, impossibile calcolare le performance.\n");
        free(node);
        return NULL; // Termina la funzione in caso di errore
    }

    double average_time = sum / count;
    const double flops = 2.0 * matrix_data->nz / average_time;
    const double mflops = flops / 1e6;

    // Inizializza i valori del nodo
    strncpy(node->nameMatrix, matrix_name, sizeof(node->nameMatrix) - 1);
    node->nameMatrix[sizeof(node->nameMatrix) - 1] = '\0';
    node->avarangeFlops = flops;
    node->avarangeMegaFlops = mflops;
    node->avarangeSeconds = average_time;
    node->nextNode = NULL;

    return node;
}

/* Funzione per il preprocessamento delle matrici in input da file */
void preprocess_matrix(struct matrixData *matrix_data, int i) {
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s%s", base_path, matrix_names[i]);

    FILE *f = fopen(full_path, "r");
    if (!f) {
        perror("Errore nell'apertura del file");
        exit(EXIT_FAILURE);
    }

    /* Lettura dell'intestazione (banner) del file Matrix Market */
    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "Errore nella lettura del banner Matrix Market.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    /* Verifica del formato della matrice */
    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode)) {
        fprintf(stderr, "Il file non è in formato matrice sparsa a coordinate.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    /* Lettura dei parametri dimensionali della matrice */
    int M, N, nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        fprintf(stderr, "Errore nella lettura delle dimensioni della matrice.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    matrix_data->row_indices = malloc(nz * sizeof(int));
    matrix_data->col_indices = malloc(nz * sizeof(int));
    matrix_data->values = malloc(nz * sizeof(double));
    matrix_data->M = M;
    matrix_data->N = N;
    matrix_data->nz = nz;

    if (matrix_data->row_indices == NULL || matrix_data->col_indices == NULL || matrix_data->values == NULL || matrix_data->M == 0 || matrix_data->N == 0 || matrix_data->nz == 0) {
        perror("Errore nell'allocazione della memoria.");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < nz; j++) {
        int result;
        double value = 1.0; // Valore predefinito per matrici "pattern"

        if (mm_is_pattern(matcode)) {   // Preprocessamento matrice in formato pattern
            result = fscanf(f, "%d %d", &matrix_data->row_indices[j], &matrix_data->col_indices[j]);
        } else {
            result = fscanf(f, "%d %d %lf", &matrix_data->row_indices[j], &matrix_data->col_indices[j], &value);
        }

        if (result != (mm_is_pattern(matcode) ? 2 : 3)) {
            fprintf(stderr, "Errore nella lettura degli elementi della matrice.\n");
            free(matrix_data->row_indices);
            free(matrix_data->col_indices);
            free(matrix_data->values);
            fclose(f);
            exit(EXIT_FAILURE);
        }

        matrix_data->row_indices[j]--; // Converti a indice 0-based
        matrix_data->col_indices[j]--; // Converti a indice 0-based
        matrix_data->values[j] = value;
    }

    /* Preprocessamento matrice simmetrica */
    if (mm_is_symmetric(matcode)) {
        int extra_nz = 0;
        for (int j = 0; j < nz; j++) {
            if (matrix_data->row_indices[j] != matrix_data->col_indices[j]) {
                extra_nz++;
            }
        }

        // Estensione degli array con il numero di non zeri da aggiungere
        matrix_data->row_indices = realloc(matrix_data->row_indices, (nz + extra_nz) * sizeof(int));
        matrix_data->col_indices = realloc(matrix_data->col_indices, (nz + extra_nz) * sizeof(int));
        matrix_data->values = realloc(matrix_data->values, (nz + extra_nz) * sizeof(double));

        if (matrix_data->row_indices == NULL || matrix_data->col_indices == NULL || matrix_data->values == NULL) {
            perror("Errore nell'allocazione della memoria.");
            fclose(f);
            exit(EXIT_FAILURE);
        }

        // Aggiunta degli elementi simmetrici
        int index = nz;
        for (int j = 0; j < nz; j++) {
            if (matrix_data->row_indices[j] != matrix_data->col_indices[j]) {
                matrix_data->row_indices[index] = matrix_data->col_indices[j];
                matrix_data->col_indices[index] = matrix_data->row_indices[j];
                matrix_data->values[index] = matrix_data->values[j];
                index++;
            }
        }
        matrix_data->nz += extra_nz;
    }
    fclose(f);
}


int main() {
    const int num_matrices = sizeof(matrix_names) / sizeof(matrix_names[0]); // Numero di matrici

    struct matrixData *matrix_data = malloc(sizeof(struct matrixData));
    if (matrix_data == NULL) {
        perror("Errore nell'allocazione della memoria per matrix_data.");
        return EXIT_FAILURE;
    }

    struct matrixResultSerial *results = NULL; // Puntatore alla lista dei risultati
    struct matrixResultSerial *lastNode = NULL; // Puntatore all'ultimo nodo della lista

    struct matrixResultSerialFINAL *resultsFinalSerial = NULL;
    struct matrixResultSerialFINAL *lastNodeFinalSerial = NULL;

    for (int i = 0; i < num_matrices; i++) {
        printf("Calcolo su matrice: %s\n", matrix_names[i]);
        for (int j = 0; j < ITERATION_PER_MATRIX; j++) {
            /* Creazione di un nuovo nodo */
            struct matrixResultSerial *newNode = malloc(sizeof(struct matrixResultSerial));
            if (newNode == NULL) {
                printf("Errore: Memoria non disponibile!\n");
                exit(1);
            }

            /* Copia il nome della matrice nella struttura */
            strncpy(newNode->nameMatrix, matrix_names[i], sizeof(newNode->nameMatrix) - 1);
            newNode->nameMatrix[sizeof(newNode->nameMatrix) - 1] = '\0'; // Assicurati che la stringa sia terminata con '\0'
            newNode->next = NULL; // L'ultimo nodo punta a NULL

            preprocess_matrix(matrix_data, i);

            /* Inizializzazione del vettore di input x */
            double *x = malloc((size_t)matrix_data->N * sizeof(double));
            if (x == NULL) {
                perror("Errore nell'allocazione della memoria per il vettore x.");
                free(matrix_data);
                return EXIT_FAILURE;
            }

            /* Assegnazione delle componenti del vettore x */
            for (int j = 0; j < matrix_data->N; j++)
                x[j] = 1.0;

            /* Esecuzione del prodotto matrice vettore con formato csr in modo serializzato */
            struct matrixPerformance matrixPerformance1 = serial_csr(matrix_data->M, matrix_data->nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, x);

            /* Esecuzione del prodotto matrice vettore con formato csr con OpenMP */
            /* Esecuzione del prodotto matrice vettore con formato csr con CUDA */
            /* Esecuzione del prodotto matrice vettore con formato hll con OpenMP */
            /* Esecuzione del prodotto matrice vettore con formato hll con CUDA */

            /* Memorizzazione dei risultati nella struttura dei risultati */
            newNode->seconds = matrixPerformance1.seconds;

            /* Gestione della lista collegata */
            if (results == NULL) {
                results = newNode; // Primo nodo della lista
            } else {
                lastNode->next = newNode; // Collega l'ultimo nodo al nuovo nodo
            }
            lastNode = newNode; // Aggiorna il puntatore all'ultimo nodo

            free(x);
        }

        struct matrixResultSerialFINAL *newNodeFinal = calculate_performance(results, matrix_data, matrix_names[i]);

        if (resultsFinalSerial == NULL) {
            resultsFinalSerial = newNodeFinal; // Primo nodo della lista
        } else {
           lastNodeFinalSerial->nextNode = newNodeFinal; // Collega l'ultimo nodo al nuovo nodo
        }
        lastNodeFinalSerial = newNodeFinal; // Aggiorna il puntatore all'ultimo nodo

        clean_matrix_mem(matrix_data);
    }
    print_results(resultsFinalSerial);

    return EXIT_SUCCESS;
}