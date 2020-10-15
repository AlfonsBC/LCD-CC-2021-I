#include <pthread.h>
#include <stdio.h>
void *codigo_del_hilo(void *id) // función ejecutada por hilo
{
 int i;
 for(i = 0; i < 50; i++)
  printf("\n Soy el hilo: %d, iter = %d", *(int*)id, i);
  pthread_exit(id);
}
int main(void)
{
  pthread_t hilo1, hilo2; //id para hilo 1 y 2 
  int id1 = 11; // id artificial para hilo 1
  int id2 = 55; // id artificial para hilo 2
  //int *salida;
  pthread_create(&hilo1, NULL, codigo_del_hilo, &id1); // creación hilo 1 
  pthread_create(&hilo2, NULL, codigo_del_hilo, &id2); // creción hilo 2
  pthread_join(hilo1, NULL);
  pthread_join(hilo2, NULL);
  printf("\n Hilos terminados \n");
  return(0);
}