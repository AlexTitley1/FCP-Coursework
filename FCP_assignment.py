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
        self.index = number
        self.connections = connections
        self.value = value


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
        else:
            self.nodes = nodes

    def get_mean_degree(self):
        '''
        Calculates the mean degree of a network
        '''
        total_degree = sum(sum(node.connections) for node in self.nodes) #calculates the total degree of the network by summing up the connections of all nodes
        return total_degree / len(self.nodes)
        pass

    def get_mean_clustering(self):
        '''
        Calculates the mean clustering coefficient of the network.
        '''
        total_cc = 0
        for node in self.nodes:
            neighbours = [self.nodes[i] for i, conn in enumerate(node.connections) if conn]
            num_neighbours = len(neighbours)
            if num_neighbours <= 2:
                continue # skip if there is less than two neighbours
            possible_triangles = num_neighbours * (num_neighbours -1)/2
            actual_triangles = 0
            for i in range(1,num_neighbours):
                for j in range(i + 1, num_neighbours):
                        if node.connections[neighbours[i].index] and node.connections[neighbours[j].index]:
                            actual_triangles += 1
            cc = actual_triangles / possible_triangles if possible_triangles !=0 else 0
            total_cc += cc
        return total_cc / len(self.nodes)       
        pass

    def get_mean_path_length(self):
        '''
        Calculates the mean path length of the network.
        '''
        total_path_length = 0
        total_pairs = 0
        for node in self.nodes:
            distances = self.bfs(node)
            total_path_length += sum(distances.values())
        return total_path_length / (len(self.nodes)*(len(self.nodes)-1))       
        pass

    def bfs(self, start_node):
        '''
        This function conducts a breadth first search from a start node.
        '''
        distances = {node.index: float('inf') for node in self.nodes} # initialises the distances dictionary with all nodes having an infinite distance from the start node
        distances[start_node.index] = 0
        queue = [start_node]
        while queue:
            current_node = queue.pop()
            for neighbour_index, conn in enumerate(current_node.connections):
                if conn and distances[neighbour_index] == float('inf'): # updates the distance to a neighbour node if there is a connection and the distance is not yet calculated
                    distances[neighbour_index] = distances[current_node.index] + 1
                    queue.append(self.nodes[neighbour_index])
            for node in self.nodes:
                if node.connections[current_node.index] and distances[node.index] == float('inf'):
                    distances[node.index] = distances[current_node.index] = 1
                    queue.append(node)        
        return distances
        
    def make_random_network(self, N, connection_probability):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random() # generates a random float between 0 and 1 to represent the value associated with a node
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))
            
        # ensures that every node has at least one connection to another node
        for (index, node) in enumerate(self.nodes):
            other_node_index = random.choice([i for i in range(N) if i !=index])
            node.connections[other_node_index] = 1
            self.nodes[other_node_index].connections[index] = 1

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability: 
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1 

        mean_degree = self.get_mean_degree()
        mean_clustering_coefficient = self.get_mean_clustering()
        mean_path_length = self.get_mean_path_length()
                  
        print("Mean degree:", mean_degree)
        print("Mean clustering coefficient:", mean_clustering_coefficient)
        print("Mean path length:",mean_path_length)

        node_coordinates = {node: (np.random.uniform(0, N), np.random.uniform(0, N)) for node in range(N)}

        plt.figure()
        for node in range (N):
            x1, y1 = node_coordinates[node]
            plt.plot(x1, y1,'o', color='black') # plot nodes
            for neighbour_index, conn in enumerate(self.nodes[node].connections):
                if conn:
                    x2, y2 = node_coordinates[neighbour_index]
                    plt.plot([x1,x2], [y1,y2],'-', color='black')
        plt.title("Random Network")
        plt.show()
    def make_ring_network(self, N, neighbour_range=1):
       self.nodes = [Node(0, x) for x in range(N)]
        for node in self.nodes:
            node.connections = [0 for _ in range(0,N)]
            for neighbour_index in range((neighbour_range * -1), neighbour_range + 1):
                if neighbour_index != 0:
                    node.connections[((node.index + neighbour_index)%N)] = 1

    def make_small_world_network(self, N, re_wire_prob=0.2):
        self.make_ring_network(N,2)
        edges = []
        for node in self.nodes:
            for neighbour_pos in range(node.index,N):
                if node.connections[neighbour_pos] == 1:
                    edges.append((node.index,neighbour_pos))
        for edge in edges:
            if np.random.random() <= re_wire_prob:
                #print("Pass")
                start_node,end_node = self.nodes[edge[0]], self.nodes[edge[1]]
                combined_connections = [a or b for a, b in zip(start_node.connections, end_node.connections)]
                possible_rewires = [i for i in range(0,N) if combined_connections[i] == 0]
                try:
                    chosen_rewire = possible_rewires[np.random.randint(0,len(possible_rewires)+1)]
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

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''
import numpy as np
import matplotlib.pyplot as plt
import argparse


def calculate_agreement(x, y, grid,H):
    total_agreement = 0
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
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
    assert(calculate_agreement(1,1, population, H_test)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(1,1, population, H_test)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(1,1, population, H_test)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(1,1, population, H_test)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(1,1, population, H_test)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(1,1, population, H_test)==4), "Test 6"

    H_test = 1

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(1,1, population, H_test)== 3), "Test 7"
    assert(calculate_agreement(1,1, population, -1)==5), "Test 8"
    assert (calculate_agreement(1,1, population, 10) == 14), "Test 9"
    assert (calculate_agreement(1,1, population, -10) == -6), "Test 10"

    print("Tests passed")


def ising_main(population, alpha,H):
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

def main():
    parser = argparse.ArgumentParser(description='Run the ising model simulations using settings.')
    parser.add_argument('--ising_model', action='store_true', help='Run the ising model simulation using default settings.')
    parser.add_argument('--external', type=float, default=0.0, help='Set the strength of the external influence on the model.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Set the alpha value used in the agreement calculation.')
    parser.add_argument('--test_ising', action='store_true', help='Run the ising model test functions to ensure integrity.')

    # Process the provided data
    args = parser.parse_args()

    if args.alpha <= 0:
        parser.error("The alpha parameter must be greater than 0.")

    if args.test_ising:
        test_ising()
    elif args.ising_model:
        population = np.random.choice([-1, 1], size=(100, 100))
        ising_main(population, args.alpha, args.external)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()


    return np.random * population





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
    defuant(0.5,0.5)
    defuant(0.1,0.5)
    defuant(0.5,0.1)
    defuant(0.1,0.2)
    plt.show()   





'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import argparse
import networkx as nx  # Import networkx to create and manipulate complex networks

class Node:
    def __init__(self, value, index, connections=None):
        """Initialize a node with a value, an index, and an optional list of connections."""
        self.value = value
        self.index = index
        self.connections = connections if connections is not None else []

class Network:
    def __init__(self, nodes=None):
        """Initialize a network with an optional list of nodes."""
        self.nodes = nodes if nodes is not None else []

    def make_random_network(self, N, connection_probability):
        """Generate a random network with N nodes and a given probability of connection between nodes."""
        self.nodes = [Node(np.random.random(), i, []) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if np.random.random() < connection_probability:
                    self.nodes[i].connections.append(j)
                    self.nodes[j].connections.append(i)

    def make_small_world_network(self, N, rewiring_prob):
        """Create a small world network with N nodes and a specific rewiring probability."""
        G = nx.watts_strogatz_graph(N, k=4, p=rewiring_prob)  # Use Watts-Strogatz model
        self.nodes = [Node(np.random.choice([-1, 1]), i, []) for i in range(N)]
        for i, j in G.edges():
            self.nodes[i].connections.append(j)
            self.nodes[j].connections.append(i)

    def ising_model_update(self):
        """Update node values based on the Ising model: nodes take the sign of the sum of their neighbors' values."""
        for node in self.nodes:
            total_influence = sum(self.nodes[i].value for i in node.connections)
            node.value = 1 if total_influence >= 0 else -1

    def deffuant_model_update(self, threshold=0.3):
        """Update node values based on the Deffuant model: nodes average their values if their opinions are close enough."""
        for node in self.nodes:
            for neighbor in node.connections:
                if abs(node.value - self.nodes[neighbor].value) < threshold:
                    mid_value = (node.value + self.nodes[neighbor].value) / 2
                    node.value = mid_value
                    self.nodes[neighbor].value = mid_value

def plot_network(network, model='ising', num_frames=50, interval=200):
    """Plot the network and animate it using the specified model."""
    fig, ax = plt.subplots()
    num_nodes = len(network.nodes)
    radius = 10  # Define the radius for laying out nodes in a circle

    # Position nodes in a circle for better visualization
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    # Initialize plot elements
    node_plot = ax.scatter(x, y, c=[node.value for node in network.nodes], cmap='viridis', s=100)
    lines = []
    for i, node in enumerate(network.nodes):
        for conn_index in node.connections:
            line, = ax.plot([x[i], x[conn_index]], [y[i], y[conn_index]], '-', color='black', alpha=0.5)
            lines.append(line)

    circle = patches.Circle((0, 0), radius, edgecolor='orange', facecolor='none', linewidth=2)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.axis('off')

    def update(frame):
        """Update function for animation, applies model updates to node values."""
        if model == 'ising':
            network.ising_model_update()
        elif model == 'deffuant':
            network.deffuant_model_update()
        node_plot.set_array(np.array([node.value for node in network.nodes]))
        return node_plot, *lines

    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
    plt.show()

def main():
    """Main function to handle argument parsing and execute the network simulation."""
    parser = argparse.ArgumentParser(description='Run Ising or Deffuant model on a network.')
    parser.add_argument('-use_network', type=int, default=10, help='Size of the network')
    parser.add_argument('-ising_model', action='store_true', help='Run Ising model')
    parser.add_argument('-deffuant', action='store_true

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-beta', type=float, default=0.2, help='Coupling coefficient')
    parser.add_argument('-threshold', type=float, default=0.2, help='Opinion difference threshold')
    parser.add_argument('-defuant', action='store_true', help='Run the Deffuant model')

    args = parser.parse_args()

    if args.defuant:
        defuant(args.beta, args.threshold)


if __name__ == '__main__':
    main()

   
