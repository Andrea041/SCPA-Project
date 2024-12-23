#include <stdio.h>
#include <omp.h>

int main() {
    const int N = 10; // Dimensione del lavoro
    int i;

    // Esecuzione parallela
#pragma omp parallel for
    for (i = 0; i < N; i++) {
        int thread_id = omp_get_thread_num(); // Ottieni l'ID del thread corrente
        printf("Thread ID: %d sta eseguendo l'iterazione %d\n", thread_id, i);
    }

    return 0;
}
