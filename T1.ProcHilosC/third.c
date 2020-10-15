#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NUM_HILOS 5
int I = 0; // variable global que puede ser modificada por todos los hilos
void *codigo_del_hilo(void *id) // codigo ejecutado por cada hilo
{
    if(*(int*)id == 3){pthread_exit(id);}
    int i;
    for(i = 0; i < 1; i++)
      printf("Hilo %d; i = %d, I = %d\n", *(int*)id, i, I++);
    pthread_exit(id);
}
int main(void) 
{
    int h;
    pthread_t hilos[NUM_HILOS];
    int id[NUM_HILOS] = {1, 2, 3, 4, 5}; // inicializacion id de hilos
    int error;
    int *salida;
    for (h = 0; h < NUM_HILOS; h++){ // loop para crear todos los hilos
        error = pthread_create(&hilos[h], NULL, codigo_del_hilo, &id[h]);
        if (error){
          fprintf(stderr, "Error %d %s\n", error, strerror (error));
          exit(-1);
        }
    }
    for (h = 0; h < NUM_HILOS; h++){ // loop para esperar que termine todos los hilos
        error = pthread_join(hilos[h], (void **)&salida);
        if (error)
          fprintf(stderr, "Error %d %s\n", error, strerror (error));
        else
          printf("hilo %d terminado\n", *salida);
    }
}