import numpy as np
import networkx as nx  
import time 
import matplotlib.pyplot as plt  
plt.style.use("seaborn")
from string import ascii_letters as abc 
abc = [x for x in abc]
from collections import defaultdict 

import base64
import requests

from multiprocessing import Pool

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

        return number_of_edges_twice//2 - counter//2

def parent_selection(input_population, number_of_pairs, method='FPS'):
    input_n = len(input_population)

    if method == 'FPS': 
        fitness_sum = sum([person.fitness for person in input_population])
        probabilities = np.array([person.fitness / fitness_sum for person in input_population])

        I_x = np.random.choice(np.arange(0, input_n), number_of_pairs, p=probabilities)
        I_y = np.random.choice(np.arange(0, input_n), number_of_pairs, p=probabilities)

        return [(input_population[I_x[i]], input_population[I_y[i]]) for i in range(number_of_pairs)]

def generate_random_graph_from_G(G):
    adj = []
    for i in range(len(G.nodes)):
        try: adj.append(list(G.adj[i].keys()))
        except: adj.append([])
    return G.number_of_edges(), adj

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

def population_update(input_population, percentage_to_keep=0.1, genetic_op='mutation', n_colors=3):
    input_population_size = len(input_population)
    output_population = []

    input_population.sort(key=lambda x: x.fitness, reverse=True)
    output_population += input_population[:int(input_population_size * percentage_to_keep)]

    list_of_parent_pairs = parent_selection(input_population, input_population_size//2)
    childs = []
    pair_index = 0
    while len(output_population)+len(childs) < input_population_size:
        child_1, child_2 = genetic_operator(list_of_parent_pairs[pair_index], method=genetic_op, n_colors=n_colors)
        childs.append(child_1); childs.append(child_2)
        pair_index += 1
    
    output_population += childs 
    
    return output_population

def generate_random_initial_population(population_size, n_nodes, al, n_colors):
    # population_size, n_nodes, al, n_colors = parameters
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

def evolution(input_population, n_generations, percentage_to_keep, genetic_op, n_colors):
    # input_population, n_generations, population_size, percentage_to_keep=0.1, genetic_op='mutation', n_colors=3
    # input_population, n_generations, percentage_to_keep, genetic_op, n_colors = parameters
    for i in range(n_generations):
        fitness_list, ix, fittest_coloring = find_fittest(input_population)
        genetic_op = np.random.choice(['SPC', 'mutation'], 1)[0]
        output_population = population_update(input_population, percentage_to_keep=percentage_to_keep, genetic_op=genetic_op, n_colors=n_colors)
        input_population = output_population
    print(fittest_coloring.fitness)
    return output_population

def coordinate(population_size, n_generations, percentage_of_parents_to_keep, n_nodes, al, n_colors, n_workers):
    islands_params = [(population_size//n_workers, n_nodes, al, n_colors)]*n_workers
    with Pool(n_workers) as p:
        islands = p.starmap(generate_random_initial_population, islands_params)
    
    new_islands_params = [(island, n_generations, percentage_of_parents_to_keep, 'mutation', n_colors) for island in islands]
    with Pool(n_workers) as p:
        new_islands = p.starmap(evolution, new_islands_params)

    return new_islands 

def visualize_results(results_fitness, results_fittest, number_of_generations_to_visualize=6):
    y = [x.fitness for x in results_fittest]
    plt.plot(range(len(results_fittest)), y)

def get_graph(name):
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
    return n_nodes, n_edges, n_colors, adj 

if __name__ == '__main__':
    t1 = time.time()
    n_nodes, n_edges, n_colors, adj = get_graph('david')
    population_size = 750
    n_generations = 4000
    percentage_of_parents_to_keep = 0.3
    n_workers = 15
    print("Number of edges: " + str(n_edges))
    coordinate(population_size, n_generations, percentage_of_parents_to_keep, n_nodes, adj, n_colors, n_workers)
    t2 = time.time()
    total = t2-t1
    print(f"{total: .2f}")