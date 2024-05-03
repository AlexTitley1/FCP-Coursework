import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import math
import argparse


class Node:
    '''
    Class representing a node in the network.
    '''

    def __init__(self, value, number, connections=None):
        '''
        Function initilises a node.
        value: the value associated with the node
        number: index of the node
        connections: list representing connections to other nodes, the default is None
        '''
        self.index = index
        self.connections = []
        self.value = random.choice([-1, 1])  # Ising model spin initialization


class Network:
    '''
    Class representing a network of nodes.
    '''

    def __init__(self, nodes=None):
        '''
        Function initialises a network.

        Args:
        nodes: list of nodes in the network, the default is None
        '''

        if nodes is None:
            self.nodes = []
            self.N = 0

        else:
            self.nodes = nodes
            self.N = len(nodes)


    def get_mean_degree(self):
        '''
        Calculates the mean degree of a network
        '''
        total_degree = sum(sum(node.connections) for node in self.nodes)
        # Calculates the total degree of the network by summing up the connections of all nodes.
        return total_degree / len(self.nodes)
        pass

    def get_mean_clustering(self):
        '''
        Calculates the mean clustering coefficient of the network.
        '''

        total_cc = 0
        for node in self.nodes:
            neighbours = [self.nodes[i] for i, conn in enumerate(node.connections) if conn == 1]
            # Finds the neighbours of the current node by checking the connections of each node.
            num_neighbours = len(neighbours)
            if num_neighbours <= 2:
                continue  # Skip if there are less than two neighbours.
            possible_triangles = num_neighbours * (num_neighbours - 1) / 2
            actual_triangles = 0
            for i in range(0, self.N):
                for j in range(i + 1, self.N):
                    if node.connections[i] and node.connections[j]:
                        actual_triangles += 1
            cc = actual_triangles / possible_triangles if possible_triangles != 0 else 0
            # Iterates over all pairs of nodes and counts and count the number of triangles that include the current node.
            total_cc += cc
        return total_cc / len(self.nodes)


def get_mean_path_length(self):
    '''
    Calculates the mean path length of the network.
    '''
    distances = []
    for node in self.nodes:
        distances.extend(self.bfs(node))
        # Uses the breadth first search to find the shortest paths from the current node to all other nodes.
        # Distances are then added to the list 'distances'.
    return round(int(sum(distances)) / (len(self.nodes) * (len(self.nodes) - 1)), 15)


def bfs(self, start_node):
    visitited = set()
    queue = [start_node.index]
    distances = np.full((self.N, 1), np.inf)
    distances[start_node.index] = 0

    while queue != []:
        # Loop until the queu is empty.
        current_node = queue.pop(0)
        visitited.add(current_node)
        # Remove the first node from the queue and mark as visited.
        temp = self.nodes[current_node]
        neighbours = [x for x, value in enumerate(temp.connections) if value == 1]
        for x in neighbours:
            if x not in visitited:
                queue.append(x)
                distances[x] = min(distances[current_node] + 1, distances[x])

    return distances
    # Returns an array of distances from the starting node to all other nodes.


def make_random_network(self, N, connection_probability):
    '''
    This function makes a *random* network of size N.
    Each node is connected to each other node with probability p
    '''

    self.nodes = []
    for node_number in range(N):
        value = np.random.random()
        # Generates a random float between 0 and 1 to represent the value associated with a node.
        connections = [0 for _ in range(N)]
        self.nodes.append(Node(value, node_number, connections))

    for (index, node) in enumerate(self.nodes):
        other_node_index = random.choice([i for i in range(N) if i != index])
        node.connections[other_node_index] = 1
        self.nodes[other_node_index].connections[index] = 1
        # Ensures that every node has at least one connection to another node.

    for (index, node) in enumerate(self.nodes):
        for neighbour_index in range(index + 1, N):
            if np.random.random() < connection_probability:
                node.connections[neighbour_index] = 1
                self.nodes[neighbour_index].connections[index] = 1

    mean_degree = self.get_mean_degree()
    mean_clustering_coefficient = self.get_mean_clustering()
    mean_path_length = self.get_mean_path_length()

    print("Mean degree:", mean_degree)
    print("Mean clustering coefficient:", mean_clustering_coefficient)
    print("Mean path length:", mean_path_length)

    node_coordinates = {node: (np.random.uniform(0, N), np.random.uniform(0, N)) for node in range(N)}
    # Assigns a random position to each node in the network.

    plt.figure()
    for node in range(N):
        x1, y1 = node_coordinates[node]
        plt.plot(x1, y1, 'o', color='black')  # plot nodes
        for neighbour_index, conn in enumerate(self.nodes[node].connections):
            if conn:
                x2, y2 = node_coordinates[neighbour_index]
                plt.plot([x1, x2], [y1, y2], '-', color='black')
    plt.title("Random Network")
    plt.show()


def make_ring_network(self, N, neighbour_range=1):
    self.nodes = [Node(np.random.random, x) for x in range(N)]
    for node in self.nodes:
        node.connections = [0 for _ in range(0, N)]
        for neighbour_index in range((neighbour_range * -1), neighbour_range + 1):
            if neighbour_index != 0:
                node.connections[((node.index + neighbour_index) % N)] = 1


def make_small_world_network(self, N, re_wire_prob=0.2):
    self.make_ring_network(N, 2)
    edges = []
    for node in self.nodes:
        for neighbour_pos in range(node.index, N):
            if node.connections[neighbour_pos] == 1:
                edges.append((node.index, neighbour_pos))
    for edge in edges:
        if np.random.random() <= re_wire_prob:
            # print("Pass")
            start_node, end_node = self.nodes[edge[0]], self.nodes[edge[1]]
            combined_connections = [a or b for a, b in zip(start_node.connections, end_node.connections)]
            possible_rewires = [i for i in range(0, N) if combined_connections[i] == 0]
            try:
                chosen_rewire = possible_rewires[np.random.randint(0, len(possible_rewires) + 1)]
                new_end_node = self.nodes[chosen_rewire]
                start_node.connections[edge[1]] = 0
                end_node.connections[edge[0]] = 0
                start_node.connections[chosen_rewire] = 1
                new_end_node.connections[edge[0]] = 1
            except:
                pass


def plot(self):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    num_nodes = len(self.nodes)
    network_radius = num_nodes * 10
    ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
    ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

    for (i, node) in enumerate(self.nodes):
        node_angle = i * 2 * np.pi / num_nodes
        node_x = network_radius * np.cos(node_angle)
        node_y = network_radius * np.sin(node_angle)

        circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
        ax.add_patch(circle)

        for neighbour_index in range(i + 1, num_nodes):
            if node.connections[neighbour_index]:
                neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                neighbour_x = network_radius * np.cos(neighbour_angle)
                neighbour_y = network_radius * np.sin(neighbour_angle)

                ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')


def test_network():
    # Ring network
    nodes = []
    num_nodes = 10
    self.N = num_nodes
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number - 1) % num_nodes] = 1
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert (network.get_mean_degree() == 2), network.get_mean_degree()
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 2.777777777777778), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert (network.get_mean_degree() == 1), network.get_mean_degree()
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 5), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert (network.get_mean_degree() == num_nodes - 1), network.get_mean_degree()
    assert (network.get_mean_clustering() == 1), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 1), network.get_mean_path_length()

    print("All tests passed")


    def visualize(self):
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.axis('off')
        angles = np.linspace(0, 2 * np.pi, self.size, endpoint=False)
        x, y = np.cos(angles), np.sin(angles)
        colors = ['red' if node.value == -1 else 'blue' for node in self.nodes]
        for i, node in enumerate(self.nodes):
            for conn in node.connections:
                plt.plot([x[i], x[conn]], [y[i], y[conn]], 'k-', alpha=0.3)
            plt.scatter(x[i], y[i], c=colors[i], s=100)
        plt.show()


'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''


def calculate_agreement(x, y, grid, H):
    '''
    Calculates the total consistency of a cell with its neighbouring cells, adjusted by the external field H.
    '''

    total_agreement = 0
    neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    for nx, ny in neighbors:
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            total_agreement += grid[x, y] * grid[nx, ny]
    total_agreement += H * grid[x, y]
    return total_agreement


def ising_step(grid, alpha, H):
    x, y = np.random.randint(0, len(grid)), np.random.randint(0, len(grid[0]))
    Di = calculate_agreement(x, y, grid, H)

    if Di < 0 or np.random.random() < np.exp(-Di / alpha):
        grid[x, y] *= -1


def plot_ising(im, population):
    # Update the image data
    new_im = np.array([[-1 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    # Pause to update the image
    plt.pause(0.1)


def test_ising():
    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    H_test = 0
    assert (calculate_agreement(1, 1, population, H_test) == 4), "Test 1"

    population[1, 1] = 1.
    assert (calculate_agreement(1, 1, population, H_test) == -4), "Test 2"

    population[0, 1] = 1.
    assert (calculate_agreement(1, 1, population, H_test) == -2), "Test 3"

    population[1, 0] = 1.
    assert (calculate_agreement(1, 1, population, H_test) == 0), "Test 4"

    population[2, 1] = 1.
    assert (calculate_agreement(1, 1, population, H_test) == 2), "Test 5"

    population[1, 2] = 1.
    assert (calculate_agreement(1, 1, population, H_test) == 4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert (calculate_agreement(1, 1, population, 1) == 3), "Test 7"
    assert (calculate_agreement(1, 1, population, -1) == 5), "Test 8"
    assert (calculate_agreement(1, 1, population, 10) == -6), "Test 9"
    assert (calculate_agreement(1, 1, population, -2) == 6), "Test 10"

    print("Tests passed")


def ising_main(population, alpha, H):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    for frame in range(100):
        for step in range(1000):
            ising_step(population, alpha, H)
        plot_ising(im, population)
        plt.pause(0.001)
    plt.ioff()
    plt.show()


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''
MAX_PERSON = 100
MAX_TIME = 100


def defuant(beta, threshold):
    # Initializes the comment list
    opinion = [random.uniform(0, 1) for _ in range(MAX_PERSON)]
    # Create two subgraphs
    fig, axe = plt.subplots(1, 2)
    # Set the title of the diagram
    fig.suptitle(f'Coupling: {beta}, Threshold: {threshold}')

    for i in range(MAX_TIME):
        for j in range(MAX_PERSON):
            # Select an individual A at random
            A = random.randint(0, MAX_PERSON - 1)
            B = random.choice([A - 1, A + 1]) if 0 < A < MAX_PERSON - 1 else (
                random.choice([1, MAX_PERSON - 1]) if A == 0 else random.choice([0, MAX_PERSON - 2]))
            if abs(opinion[A] - opinion[B]) <= threshold:
                oA, oB = opinion[A], opinion[B]
                opinion[A] = oA + beta * (oB - oA)
                opinion[B] = oB + beta * (oA - oB)
        # Plot the opinions of all individuals at the current time step on the scatter plot
        axe[1].scatter([i] * MAX_PERSON, opinion, c='red')
        axe[1].set_ylabel('Opinion')
    # Plot the opinion distribution on the histogram
    axe[0].hist(opinion, bins=10, color='blue')
    axe[0].set_xlabel('Opinion')
    plt.show()


def test_defuant():
    defuant(0.5, 0.5)
    defuant(0.1, 0.5)
    defuant(0.5, 0.1)
    defuant(0.1, 0.2)
    plt.show()


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

# Define the Node and Network classes


'''
class Network:
    def __init__(self, size):
        self.nodes = [Node(i) for i in range(size)]
        self.size = size

    def add_edge(self, i, j):
        if i != j and j not in self.nodes[i].connections:
            self.nodes[i].connections.append(j)
            self.nodes[j].connections.append(i)

    def make_small_world(self, neighbor_k=2, rewire_prob=0.1):
        # Create ring lattice first
        for i in range(self.size):
            for offset in range(1, neighbor_k + 1):
                self.add_edge(i, (i + offset) % self.size)
                self.add_edge(i, (i - offset + self.size) % self.size)

        # Rewiring process
        for i in range(self.size):
            for offset in range(1, neighbor_k + 1):
                if random.random() < rewire_prob:
                    j = (i + offset) % self.size
                    new_j = random.randint(0, self.size - 1)
                    while new_j == i or new_j in self.nodes[i].connections:
                        new_j = random.randint(0, self.size - 1)
                    # Remove original connection and add new one
                    self.nodes[i].connections.remove(j)
                    self.nodes[j].connections.remove(i)
                    self.add_edge(i, new_j)

    def visualize(self):
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.axis('off')
        angles = np.linspace(0, 2 * np.pi, self.size, endpoint=False)
        x, y = np.cos(angles), np.sin(angles)
        colors = ['red' if node.value == -1 else 'blue' for node in self.nodes]
        for i, node in enumerate(self.nodes):
            for conn in node.connections:
                plt.plot([x[i], x[conn]], [y[i], y[conn]], 'k-', alpha=0.3)
            plt.scatter(x[i], y[i], c=colors[i], s=100)
        plt.show()
'''

def ising_model(network, steps=1000, temperature=1.0):
    for _ in range(steps):
        for node in network.nodes:
            energy_change = 2 * node.value * sum(network.nodes[neighbor].value for neighbor in node.connections)
            if energy_change < 0 or random.random() < np.exp(-energy_change / temperature):
                node.value *= -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-beta', type=float, default=0.2, help='Coupling coefficient')
    parser.add_argument('-threshold', type=float, default=0.2, help='Opinion difference threshold')
    parser.add_argument('-defuant', action='store_true', help='Run the Deffuant model')
    parser.add_argument('-ising_model', action='store_true',
                        help='Run the ising model simulation using default settings.')
    parser.add_argument('-external', type=float, default=0.0,
                        help='Set the strength of the external influence on the model.')
    parser.add_argument('-alpha', type=float, default=1.0,
                        help='Set the alpha value used in the agreement calculation.')
    parser.add_argument('-test_ising', action='store_true',
                        help='Run the ising model test functions to ensure integrity.')
    parser.add_argument("-network", type=int)
    parser.add_argument("-test_network", action="store_true")
    parser.add_argument('-use_network', type=int, default=10, help='Size of the network')
    parser.add_argument('-size', type=int, default=20, help='Number of nodes in the network')
    parser.add_argument('-steps', type=int, default=1000, help='Number of simulation steps')
    parser.add_argument('-temperature', type=float, default=2.0, help='Temperature for the Ising model')
    parser.add_argument('-neighborhood', type=int, default=2,
                        help='Each node is connected to `neighborhood` nearest neighbors')
    parser.add_argument('-rewiring_probability', type=float, default=0.1, help='Probability to rewire each edge')
    parser.add_argument('-ising_model', action='store_true', help='Run the Ising model simulation.')
    parser.add_argument('-use_network', type=int, help='Size of the network if using network model')

    args = parser.parse_args()
    net = Network()
    if args.defuant:
        defuant(args.beta, args.threshold)
    elif args.alpha <= 0:
        parser.error("The alpha parameter must be greater than 0.")
    elif args.test_ising:
        test_ising()
    elif args.ising_model:
        population = np.random.choice([-1, 1], size=(100, 100))
        ising_main(population, args.alpha, args.external)
    elif args.test_network:
        test_network()
    elif args.network:
        network = Network()
        network.make_random_network(args.network, 0.3)
    elif args.use_network:
        network = Network(args.use_network)
        network.make_small_world(args.neighborhood, args.rewiring_probability)
        ising_model(network, args.steps, args.temperature)
        network.visualize()
    elif args.ising_model:
        # Implement grid-based Ising model here if needed
        print("Grid-based Ising model not implemented.")


if __name__ == '__main__':
    main()
