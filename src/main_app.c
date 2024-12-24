#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../libs/csrSerialized.h"
#include "../libs/mmio.h"
#include "../libs/matrixLists.h"
#include "../libs/data_structure.h"

#ifdef USER_PIERFRANCESCO
const char *base_path = "/home/pierfrancesco/Desktop/matrix/";
#elif defined(USER_ANDREA)
const char *base_path = "/Users/andreaandreoli/matrix/";
#else
const char *base_path = "./matrix/"; // Valore predefinito
#endif

int iteration_ForEach_Matrix=5;


/* Funzione per stampare i risultati dalla lista */
void print_results(struct matrixResultSerialFINAL *results) {
    printf("Lista dei risultati:\n");
    struct matrixResultSerialFINAL  *current = results;
    while (current != NULL) {
        printf("Matrix: %s\n", current->nameMatrix);
        printf("  Seconds: %f\n", current->avarangeSeconds);
        printf("  FLOPS: %f\n", current->avarangeFlops);
        printf("  MFLOPS: %f\n", current->avarangeMegaFlops);
        current = current->nextNode;
    }
}


struct matrixResultSerialFINAL *calculate_performance(struct matrixResultSerial *head, struct matrixConst matrix_const, char nameCurrentMatrix[50]) {
    // Alloca dinamicamente memoria per un nuovo nodo
    struct matrixResultSerialFINAL *node = (struct matrixResultSerialFINAL *)malloc(sizeof(struct matrixResultSerialFINAL));
    if (node == NULL) {
        perror("Errore nell'allocazione della memoria per matrixResultSerialFINAL");
        exit(EXIT_FAILURE); // Termina il programma in caso di errore
    }

    printf("Inizio calcolo delle performance per la matrice: %s\n", nameCurrentMatrix);

    double somma = 0.0;
    int count = 0;
    struct matrixResultSerial *current = head;

    // Itera attraverso la lista
    while (current != NULL) {
        printf("Aggiungo seconds: %f (attuale somma: %d)\n", current->seconds, somma);
        somma += current->seconds; // Somma i secondi
        count++;                   // Conta il numero di nodi
        printf("SOMMA: %d\n", somma);
        current = current->next;   // Passa al nodo successivo
    }

    if (count == 0) {
        fprintf(stderr, "Errore: la lista è vuota, impossibile calcolare le performance.\n");
        free(node);
        return NULL; // Termina la funzione in caso di errore
    }

    double avarangeTime = (double)somma / count;
    const double flops = 2.0 * matrix_const.nz / avarangeTime;
    const double mflops = flops / 1e6;

    printf("Tempo medio (seconds): %f\n", avarangeTime);
    printf("FLOPS calcolati: %f\n", flops);
    printf("MFLOPS calcolati: %f\n", mflops);

    // Inizializza i valori del nodo
    strncpy(node->nameMatrix, nameCurrentMatrix, sizeof(node->nameMatrix) - 1);
    node->nameMatrix[sizeof(node->nameMatrix) - 1] = '\0'; // Assicurati che la stringa sia terminata con '\0'
    node->avarangeFlops = flops;
    node->avarangeMegaFlops = mflops;
    node->avarangeSeconds = avarangeTime;
    node->nextNode = NULL; // Imposta il puntatore al nodo successivo a NULL

    printf("Nodo completato: %s\n", node->nameMatrix);
    printf("Performance:\n  - FLOPS: %f\n  - MFLOPS: %f\n  - Tempo medio: %f\n",
           node->avarangeFlops, node->avarangeMegaFlops, node->avarangeSeconds);

    return node; // Restituisce il nodo allocato dinamicamente
}



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
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) { //f_corretto
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

    struct matrixResultSerial *results = NULL; // Puntatore alla lista dei risultati
    struct matrixResultSerial *lastNode = NULL; // Puntatore all'ultimo nodo della lista

    struct matrixResultSerialFINAL *resultsFinalSerial = NULL;
    struct matrixResultSerialFINAL *lastNodeFinalSerial = NULL;

    char nameCurrentMatrix[50];

    for (int i = 0; i < num_matrices; i++) {

        struct matrixConst matrix_const ;

        for (int j=0; j<iteration_ForEach_Matrix; j++) {

             // Creazione di un nuovo nodo
                struct matrixResultSerial *newNode = (struct matrixResultSerial *)malloc(sizeof(struct matrixResultSerial));
                if (newNode == NULL) {
                    printf("Errore: Memoria non disponibile!\n");
                    exit(1);
                }
                strcpy(nameCurrentMatrix, matrix_names[i]);
                // Copia il nome della matrice nella struttura
                strncpy(newNode->nameMatrix, nameCurrentMatrix, sizeof(newNode->nameMatrix) - 1);
                newNode->nameMatrix[sizeof(newNode->nameMatrix) - 1] = '\0'; // Assicurati che la stringa sia terminata con '\0'
                newNode->next = NULL; // L'ultimo nodo punta a NULL

                matrix_const = preprocess_matrix(matrix_data, i);

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

                // Chiamata della funzione serial_csr
                struct matrixPerformance matrixPerformance1 = serial_csr(matrix_const.M, matrix_const.N, matrix_const.nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, x);

                // Memorizzazione dei risultati nella struttura dei risultati
                newNode->seconds = matrixPerformance1.seconds;


                // Gestione della lista collegata
                if (results == NULL) {
                    results = newNode; // Primo nodo della lista
                } else {
                    lastNode->next = newNode; // Collega l'ultimo nodo al nuovo nodo
                }
                lastNode = newNode; // Aggiorna il puntatore all'ultimo nodo

                free(x);

        }

            struct matrixResultSerialFINAL *newNodeFinal = calculate_performance(results, matrix_const, nameCurrentMatrix);

            if (resultsFinalSerial == NULL) {
                resultsFinalSerial = newNodeFinal; // Primo nodo della lista
            } else {
               lastNodeFinalSerial->nextNode = newNodeFinal; // Collega l'ultimo nodo al nuovo nodo
            }
            lastNodeFinalSerial = newNodeFinal; // Aggiorna il puntatore all'ultimo nodo

            free(matrix_data->row_indices);
            free(matrix_data->col_indices);
            free(matrix_data->values);


    }

    print_results(resultsFinalSerial);

    return EXIT_SUCCESS;
}

