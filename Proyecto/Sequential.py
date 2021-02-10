import matplotlib.pyplot as plt  
plt.style.use("seaborn")
import networkx as nx  
import numpy as np
import time 
from multiprocessing import Pool
from string import ascii_letters as abc 
abc = [x for x in abc]
import base64
import requests
from collections import defaultdict

class Coloration:
    def __init__(self, colors, adjacency_list):
        """
        Creacion de un objeto Coloration
        :param colors: 
        :type colors: string - contiene la coloración de la gráfica, si tenemos 3 colores puede ser 'rgggbbrg...'
        :param adj: lista de adyacencia de la grafica 
        :type adj: lista de listas 
        :return: objecto Coloration
        :rtype: Coloration
        """
        self.colors = colors
        self.adjacency_list = adjacency_list
        self.n_nodes = len(self.colors)
        self.fitness = self.get_fitness(self.colors, self.adjacency_list)

    def get_fitness(self, colors, adjacency_list):
        """
        Obtiene cuantas aristas que conecten a dos vértices de distinto color genera Coloration.
        :param colors:
        :type colors: string - contiene la coloración de la gráfica, si tenemos 3 colores puede ser 'rgggbbrg...'
        :param adj: lista de adyacencia de la grafica 
        :type adj: lista de listas
        :return: fitness de Coloration
        :rtype: int  
        """
        counter = 0 
        number_of_edges_twice = 0

        for index, node_color in enumerate(colors):
            for neighbour in adjacency_list[index]:
                number_of_edges_twice += 1
                if node_color == colors[neighbour]:
                    counter += 1

        return number_of_edges_twice/2 - counter/2

def parent_selection(input_population, number_of_pairs, method='FPS'):
    """
    Forma pares de padres, favoreciendo aquellos que tengan un mejor fitnesss
    :param input_population: poblacion de la que se desea obtener pares de padres
    :type input_population: lista de coloraciones
    :param number_of_pairs: numero de pares de padres deseados
    :type number_of_pairs: int
    :param method: metodo de seleccion
    :type method: str
    :return: lista de tuplas de pares de padres 
    :rtype: lista de listas de Coloration
    """
    input_n = len(input_population)

    if method == 'FPS':  # Fitness proportional selection
        # our fitness is non-negative so we can apply a simple formula  fitness_m/sum(fitness_i)
        fitness_sum = sum([person.fitness for person in input_population])
        probabilities = np.array([person.fitness / fitness_sum for person in input_population])

        I_x = np.random.choice(np.arange(0, input_n), number_of_pairs, p=probabilities)
        I_y = np.random.choice(np.arange(0, input_n), number_of_pairs, p=probabilities)

        return [(input_population[I_x[i]], input_population[I_y[i]]) for i in range(number_of_pairs)]

def genetic_operator(pair_of_parents, method='SPC', n_colors=3):
    """
    Dados dos padres, regresa el hijo resultante de la cruza de estos
    :param pair_of_parents: pare
    :type pair_of_parents: lista que contenga dos padres
    :param method: metodo de cruza 
    :type method: str
    :param n_colors: colores con los que se desea colorear la gráfica
    :type n_colors: int 
    :return: pares de hijos
    :rtype: par de Coloration
    """
    n_nodes = pair_of_parents[0].n_nodes
    al = pair_of_parents[0].adjacency_list

    if method == 'mutation':
        node1 = np.random.randint(0, n_nodes)
        node2 = np.random.randint(0, n_nodes)

        colors = set(abc[:n_colors])
        
        child_one_colors = pair_of_parents[0].colors
        child_two_colors = pair_of_parents[1].colors

        def get_child(child_colors):
            child_colors = list(child_colors)
            for i in range(len(child_colors)):
                invalid = set()
                for j in al[i]: 
                    if child_colors[j] == child_colors[i]: invalid.add(child_colors[j])
                if invalid:
                    valid = list(colors.difference(invalid))
                    child_colors[i] = np.random.choice(valid, 1)[0]
            return "".join(child_colors) 

        child_one_colors = get_child(child_one_colors)
        child_two_colors = get_child(child_two_colors)
        return Coloration(child_one_colors, al), Coloration(child_two_colors, al)

    if method == 'SPC': 
        point = np.random.randint(0, n_nodes)

        parent_1_colors = pair_of_parents[0].colors
        parent_2_colors = pair_of_parents[1].colors

        child_one_colors = parent_1_colors[:point] + parent_2_colors[point:]
        child_two_colors = parent_2_colors[:point] + parent_1_colors[point:]

        return Coloration(child_one_colors, al), Coloration(child_two_colors, al)

def population_update(input_population, output_population_size, generation_change_method='Elitism',
                      percentage_to_keep=0.1, genetic_op='SPC', n_colors=3):
    """
    Evoluciona la población input_population en una generación
    :param input_population: poblacion de la que se desea obtener pares de padres
    :type input_population: lista de Coloration
    :param percentage_to_keep: porcentaje de individuos que se preservan para la siguiente generación
    :type percentage_to_keep: int
    :param genetic_op: metodo de cruza 
    :type genetic_op: str
    :param n_colors: colores con los que se desea colorear la gráfica
    :type n_colors: int 
    :return: población evolucionada en una generación
    :rtype: lista de Coloration 
    """
    input_population_size = len(input_population)
    output_population = []

    if generation_change_method == 'Elitism':
        # We keep the best x percent of the input population
        input_population.sort(key=lambda x: x.fitness, reverse=True)
        output_population += input_population[:int(input_population_size * percentage_to_keep)]

        list_of_parent_pairs = parent_selection(input_population, input_population_size // 2)
        childs = []
        pair_index = 0
        while len(output_population)+len(childs) < output_population_size:
            child_1, child_2 = genetic_operator(list_of_parent_pairs[pair_index], method=genetic_op, n_colors=n_colors)
            childs.append(child_1); childs.append(child_2)
            pair_index += 1
    
    output_population.extend(childs) 
    
    return output_population

def generate_random_initial_population(population_size, n_nodes, al, n_colors):
    """
    Genera una nueva población de manera aleatoria
    :param population_size: tamaño de la población
    :type population_size: int
    :param n_nodes: número de nodos
    :type n_nodes: int
    :param adj: lista de adyacencia de la gráfica que se desea colorear
    :type adj: lista de listas
    :param n_colors: colores con los que se desea colorear la gráfica
    :type n_colors: int 
    :return: población aleatoria 
    :rtype: lista de objetos Coloration
    """
    input_population = []
    colors = abc[:n_colors]

    for _ in range(population_size):
        color_list = np.random.choice(colors, n_nodes, replace=True)
        color_string = "".join(color_list)
        input_population.append(Coloration(color_string, al))

    return input_population

def find_fittest(input_population):
    """
    Encuentra Coloration con mayor fitness de input_population
    :param input_population: poblacion de la que se desea obtener pares de padres
    :type input_population: lista de coloraciones
    :return: Coloration con mayor fitness de input_population
    :rtype: Coloration
    """
    fitness_list = [person.fitness for person in input_population]
    ix = np.argmax(fitness_list)
    return fitness_list, ix, input_population[ix]

def evolution(input_population, n_generations, population_size, percentage_to_keep=0.1, genetic_op='mutation', n_colors=3):
    """
    Regresa lista con la persona con mayor fitness para cada generación
    :param input_population: poblacion de la que se desea obtener pares de padres
    :type input_population: lista de Coloration
    :param n_generations: número de generaciones a simular
    :type n_generations: int
    :param percentage_to_keep: porcentaje de individuos que se preservan para la siguiente generación
    :type percentage_to_keep: int
    :param n_colors: colores con los que se desea colorear la gráfica
    :type n_colors: int 
    :param n_edges: número de aristas en la gráfica que se desea colorear 
    :type n_edges: int 
    :returns: Lista con la persona con mayor fitness para cada generación
    :rtype: Coloration
    """
    for i in range(n_generations):
        fitness_list, ix, fittest_coloring = find_fittest(input_population)
        genetic_op = np.random.choice(['SPC', 'mutation'], 1)[0]
        output_population = population_update(input_population, population_size, percentage_to_keep=percentage_to_keep, genetic_op=genetic_op, n_colors=n_colors)
        input_population = output_population
    print(fittest_coloring.fitness)
    return 

def visualize_results(results_fitness, results_fittest, number_of_generations_to_visualize=6):
    """
    Genera visualización de generación vs fitness para cada isla 
    :param islands_fittest: contiene listas del fitness para cada isla para cada generación
    :type islands_fittest: lista de listas 
    :param n_edges: número de aristas en la gráfica que se desea colorear 
    :type n_edges: int 
    """
    y = [x.fitness for x in results_fittest]
    plt.plot(range(len(results_fittest)), y)

def get_graph(name):
    """
    Regresa número de nodos, número de airstas, número cromatico y matriz de adyacencia de la gráfica name
    :param name: nombre de la gráfica 
    :type name: str
    :rturn n_nodes, n_edges, n_colors, adj: 
    """
    doc = f"https://raw.githubusercontent.com/carloscerlira/Datasets/master/DIMACS%20Graph%20Coloring/{name}.txt"
    doc = requests.get(doc)
    raw_edges = doc.text
    raw_edges = raw_edges.split("\n")
    raw_edges = raw_edges[3:]
    _, _, n_nodes, n_edges, n_colors = raw_edges.pop(0).split(" ")
    n_nodes, n_edges, n_colors = map(int, (n_nodes, n_edges, n_colors))
    adj = [[] for i in range(n_nodes)]
    for line in raw_edges:
        line = line.strip()
        line = line.split(" ")
        line.pop(0)
        x, y = map(int, line)
        x, y = x-1, y-1
        adj[x].append(y)
    return n_nodes, n_edges//2, n_colors, adj 

if __name__ == '__main__': 
    """
    Función principal, aqui se definen todos los parámetros del modelo como population_size, n_generations    """
    t1 = time.time()
    graph_name = "myciel4"
    n_nodes, n_edges, n_colors, adj = get_graph('queen8_8')
    population_size = 100
    n_generations = 4000
    percentage_to_keep = 0.3  
    print("Graph name: ", graph_name)
    print("Number of nodes: " + str(n_nodes))
    print("Number of edges: " + str(n_edges))
    print("Number of colors: " + str(n_colors))
    n_workers = 6
    for i in range(n_workers):
        input_population = generate_random_initial_population(population_size//n_workers, n_nodes, adj, n_colors=n_colors)
        evolution(input_population, n_generations, population_size//n_workers,
                percentage_to_keep=percentage_to_keep, n_colors=n_colors)
    t2 = time.time()
    total = t2-t1
    print(f"{total: .2f}")