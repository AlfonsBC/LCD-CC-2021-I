
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>

int main(void){ """funcion principal recibe un void, parametro vacio"""
	
	int 	fd[2], nbytes; 										"""se declaran el arreglo fd y el entero nbytes"""
	pid_t	childpid; 												""" Se declara una variable que guarde el Identificador del Proceso en la variable Childpid"""
	char	cadena[] = "Hola, mundo!!\n";			""" Se declara cadena o string en una variable llamada de cadena[] en la que pueda incrementar de tamano """
	char	lectura[80];											""" Se declara una cadena en la variable lectura[80], cuya longitud maxima es de 80 """

	pipe(fd);																"""fd se convierte en un pipe que comunica el proceso padre con el proceso hijo"""

	if ((childpid = fork()) == -1){ 				"""el comando fork(), nos crea un nuevo procesp"""
		perror("fork"); """se verifica que el nuevo proceso se haya creado correctamente"""
		exit(1);				""" Una vez impreso el mensaje se sale del if """

	}

	if (childpid == 0){ 		""" Si el Identificador del Proceso es 0, """

												/* Proceso hijo cierra el pipe en el canal de entrada */
		close(fd[0]);

												/* Envía "cadena" por medio del canal de salida del pipe */
		write(fd[1], cadena, strlen(cadena));
		exit(0);						""" Una vez escrito y enviado la "cadena" se sale """
	}
	else{									""" Si el Identificador es distinto de cero """
		
		/* Proceso padre cierra el canal de salida del pipe */
		close(fd[1]);
		/* Lee "cadenacar" del pipe */
		nbytes = read(fd[0], lectura, sizeof(lectura));
		printf("Cadena recibida: %s", lectura);
	}

	return(0); """ Ya se acaba el programa y nos devuelve 0"""
	
}
