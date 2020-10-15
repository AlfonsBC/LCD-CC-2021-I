#include <pthread.h> 
#include <stdio.h>

void *codigo_del_hilo(void *id) // función que ejecutara hilo
{
 int i;
 for(i = 0; i < 50; i++)
  printf("\n Soy el hilo: %d, iter = %d", *(int*)id, i);
  pthread_exit(id);
}

int main(void)
{
  pthread_t hilo; // id para hijo
  int id = 245; // id artificial
  int *salida; 
  pthread_create(&hilo, NULL, codigo_del_hilo, &id); // creación de hilo 
  pthread_join(hilo, NULL); // espera hasta que hilo términe 
  printf("\n Hilo %d terminado \n", *salida);
  return(0);
}