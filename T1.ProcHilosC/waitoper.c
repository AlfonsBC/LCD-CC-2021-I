#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void)
{
    // inicialización de variables
    int i;
    int a,b;
    pid_t pidh1,pidh2,pidx; // variables usada para id de procesos
    int prod, mayor;
    int res;

    printf("\nDame dos enteros: \n");
    scanf("%d%d", &a, &b);
    pidh1 = fork(); // creación hijo 1

    // código del padre
    if(pidh1)
    {
        pidh2 = fork(); // creación hijo 2
        // código del padre
        if(pidh2){
            for(i = 0; i < 2; i++)
            {
                pidx = wait(&res); // padre espera a que cualquiera de los hijos termine y guarda status en res
                if(pidx == pidh1){ // si id de proceso que termino es hijo 1
                    prod = WEXITSTATUS(res);
                }
                else{ // si id de proceso que termino es hijo 2
                    mayor = WEXITSTATUS(res);
                }
            }
            // padre se encarga de imprimir los resultados obtenidos por hijo 1 e hijo 2
            printf("\n El producto de %d y %d es %d", a, b, prod); 
            printf("\n El mayor de %d y %d es %d \n", a, b, mayor);
        }
        // código del hijo 2
        else{ 
            if(a > b){
                exit(a); // hijo 1 términa y regresa a si a>b como status
            }
            else{
                exit(b); // hijo 1 términay regresa b si b>=a como status
            }
        }
    }
    // código del hijo 1
    else{
        prod = a*b;
        exit(prod); // hijo 2 termina y regresa prod como status
    }
    return(0);
 }
