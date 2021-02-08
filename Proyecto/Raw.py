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

class World_Map:
    def __init__(self, colors, adjacency_list):
        self.colors = colors
        self.adjacency_list = adjacency_list
        self.n_nodes = len(self.colors)
        self.fitness = self.get_fitness(self.colors, self.adjacency_list)

    def get_fitness(self, colors, adjacency_list):
        counter = 0 
        number_of_edges_twice = 0

        for index, node_color in enumerate(colors):
            for neighbour in adjacency_list[index]:
                number_of_edges_twice += 1
                if node_color == colors[neighbour]:
                    counter += 1

        return number_of_edges_twice/2 - counter/2

def generate_random_graph(number_of_nodes, probability_of_edge):
    G = nx.fast_gnp_random_graph(number_of_nodes, probability_of_edge, seed=None, directed=False)
    edges = []
    for i in range(number_of_nodes):
        temp1 = G.adj[i]
        edges.append(list(G.adj[i].keys()))
    return G.number_of_edges(), edges

def generate_random_graph_from_G(G):
    edges = []
    for i in range(len(G.nodes)):
        temp1 = G.adj[i]
        edges.append(list(G.adj[i].keys()))
    return G.number_of_edges(), edges

def parent_selection(input_population, number_of_pairs, method='FPS'):
    input_n = len(input_population)

    if method == 'FPS':  # Fitness proportional selection
        # our fitness is non-negative so we can apply a simple formula  fitness_m/sum(fitness_i)
        fitness_sum = sum([person.fitness for person in input_population])
        probabilities = np.array([person.fitness / fitness_sum for person in input_population])

        I_x = np.random.choice(np.arange(0, input_n), number_of_pairs, p=probabilities)
        I_y = np.random.choice(np.arange(0, input_n), number_of_pairs, p=probabilities)

        return [(input_population[I_x[i]], input_population[I_y[i]]) for i in range(number_of_pairs)]

def genetic_operator(pair_of_parents, method='SPC', n_colors=3):
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
        return World_Map(child_one_colors, al), World_Map(child_two_colors, al)

    if method == 'SPC': 
        point = np.random.randint(0, n_nodes)

        parent_1_colors = pair_of_parents[0].colors
        parent_2_colors = pair_of_parents[1].colors

        child_one_colors = parent_1_colors[:point] + parent_2_colors[point:]
        child_two_colors = parent_2_colors[:point] + parent_1_colors[point:]

        return World_Map(child_one_colors, al), World_Map(child_two_colors, al)

def population_update(input_population, output_population_size, generation_change_method='Elitism',
                      percentage_to_keep=0.1, genetic_op='SPC', n_colors=3):
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
    input_population = []
    colors = abc[:n_colors]

    for _ in range(population_size):
        color_list = np.random.choice(colors, n_nodes, replace=True)
        color_string = "".join(color_list)
        input_population.append(World_Map(color_string, al))

    # print('A random population of ' + str(population_size) + ' people was created')

    return input_population

def find_fittest(input_population):
    fitness_list = [person.fitness for person in input_population]
    ix = np.argmax(fitness_list)
    return fitness_list, ix, input_population[ix]

def evolution(input_population, n_generations, population_size, percentage_to_keep=0.1, genetic_op='mutation', n_colors=3):
    for i in range(n_generations):
        fitness_list, ix, fittest_coloring = find_fittest(input_population)
        genetic_op = np.random.choice(['SPC', 'mutation'], 1)[0]
        output_population = population_update(input_population, population_size, percentage_to_keep=percentage_to_keep, genetic_op=genetic_op, n_colors=n_colors)
        input_population = output_population
    print(fittest_coloring.fitness)
    return 

def visualize_results(results_fitness, results_fittest, number_of_generations_to_visualize=6):
    y = [x.fitness for x in results_fittest]
    plt.plot(range(len(results_fittest)), y)

def get_graph(name):
    doc = f"https://raw.githubusercontent.com/carloscerlira/Datasets/master/DIMACS%20Graph%20Coloring/{name}.txt"
    doc = requests.get(doc)
    raw_edges = doc.text
    raw_edges = raw_edges.split("\n")
    edges = []
    for line in raw_edges:
        line = line.strip()
        line = line.split(" ")
        line.pop(0)
        x, y = map(int, line)
        x, y = x-1, y-1
        edges.append((x, y))
    G = nx.Graph()
    G.add_edges_from(edges)
    return G 


if __name__ == '__main__': 
    t1 = time.time()
    G = get_graph('david')
    number_of_edges, al = generate_random_graph_from_G(G)
    population_size = 100
    n_nodes = len(al)  
    n_generations = 4000
    genetic_op = 'SPC'  
    percentage_of_parents_to_keep = 0.3  
    n_colors = 8
    print("Number of edges: " + str(number_of_edges))
    n_workers = 6
    for i in range(n_workers):
        input_population = generate_random_initial_population(population_size//n_workers, n_nodes, al, n_colors=n_colors)
        evolution(input_population, n_generations, population_size//n_workers,
                percentage_to_keep=percentage_of_parents_to_keep,
                genetic_op=genetic_op, n_colors=n_colors)
    t2 = time.time()
    total = t2-t1
    print(f"{total: .2f}")