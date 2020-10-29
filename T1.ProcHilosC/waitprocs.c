#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

# define NUM_PROCESOS 5   /*  */
int I = 0;   /* Aquí estamos declarando en iniciando la variable enter I */

void codigo_del_proceso (int id) /* Está funcion nos genera el proceso con id dado */
{
   int i;   /* Comenzamos declarando un entero i que usaremos en nuestro ciclo */

   for (i = 0; i < 50; i++) /* El ciclo hace 49 ciclos */
        printf("Proceso %d: i = %d, I = %d\n", id, i, I++ ); /*Aquí imprimimos el número de proceso,  */
   exit (id); /* Cerramos la variable para evitar alterar su valor */

}

int main(void) /* Esta es la función principal*/ 
{
    int p;
    int id[NUM_PROCESOS] = {1, 2, 3, 4, 5};
    int pid;
    int salida; /*Iniciamos las variables p, pid y salida serán enteros, y id  id tendrá en total 5 procesos  */ 

    for (p = 0; p < NUM_PROCESOS; p++) {  /* p nos servirá para recorrer cada proceso*/
      pid = fork(); /* Con fork asignamos un proceso a pid */
      if  (pid == -1){    /* Este if y else sirven como un switch, en caso de que fork cree el proceso, pid ==-1 */
          perror("Error al crear un proceso: "); /* Así que en este caso nos arroja el error y cierra el proceso */
          exit(-1);
      }
      else if (pid == 0) /* codigo proceso hijo */
          codigo_del_proceso (id[p]); /* Aquí ejecutamos la función que hicimos antes, para id = 1,2,3,5*/
    }

    // codigo proceso padre
    for (p = 0; p < NUM_PROCESOS; p++) { /* correremos p de 0 a 4 */
        pid = wait(&salida);  /*  La función wait hace esperar al proceso hasta que el proceso hijo acabe y guardará el pid del hijo en salida*/
        printf("Proceso %d con id = %x terminado\n", pid, salida >> 8); /* muestra el proceso hijo que termino, su id lo guardo en salida, desplaza la salida a 8  */
    }

    return(0);
}
