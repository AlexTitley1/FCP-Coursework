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
            if num_neighbours < 2:
                continue # skip if there is less than two neighbours
            possible_triangles = num_neighbours * (num_neighbours -1)/2
            actual_triangles = 0
            for i in range(num_neighbours):
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


def test_networks():
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
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 2.777777777777778), network.get_path_length()

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
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 5), network.get_path_length()

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
    assert (network.get_clustering() == 1), network.get_clustering()
    assert (network.get_path_length() == 1), network.get_path_length()

    print("All tests passed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-network", type=int)
    parser.add_argument("-test_network", action = "store_true")

    args = parser.parse_args()

    if args.test_network:
        test_network()
    elif args.network:
        network=Network()
        network.make_random_network(args.network, 0.3)
        

if __name__ == "__main__":
    main()

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''


def calculate_agreement(population, row, col, external=0.0):
    '''
    This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
    Inputs: population (numpy array)
            row (int)
            col (int)
            external (float)
    Returns:
            change_in_agreement (float)
    '''

    # Your code for task 1 goes here

    return np.random * population


def ising_step(population, external=0.0):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external "pull" on opinion
    '''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, external=0.0)

    if agreement < 0:
        population[row, col] *= -1


# Your code for task 1 goes here

def plot_ising(im, population):
    '''
    This function will display a plot of the Ising model
    '''


    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)


def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model
    '''


    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1) == 4), "Test 1"

    population[1, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -4), "Test 2"

    population[0, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -2), "Test 3"

    population[1, 0] = 1.
    assert (calculate_agreement(population, 1, 1) == 0), "Test 4"

    population[2, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == 2), "Test 5"

    population[1, 2] = 1.
    assert (calculate_agreement(population, 1, 1) == 4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
    assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
    assert (calculate_agreement(population, 1, 1, 10) == 14), "Test 9"
    assert (calculate_agreement(population, 1, 1, -10) == -6), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''
import matplotlib.pyplot as plt
import random
import math
import argparse
MAX_PERSON = 100
MAX_TIME = 100

def defuant(beta,threshold):
        # Initializes the comment list
    opinion = [random.uniform(0,1) for _ in range(MAX_PERSON)]
    # Create two subgraphs
    fig,axe = plt.subplots(1,2)
    # Set the title of the diagram
    fig.suptitle('Coupling:{},Threshold:{}'.format(beta,threshold))
    for i in range(MAX_TIME):
        for j in range(MAX_PERSON):
            # Select an individual A at random
            A = random.randint (0,MAX_PERSON-1)
            # Select a neighbor B of individual A
            if A == 0:
                B = 1 or 99
            elif A == 99:
                B = 98 or 0
            else:
                prob = random.uniform(0,1)
                B = A - 1 if prob <= 0.5 else A + 1
            # If the difference between the opinions of A and B is less than or equal to the threshold, the opinions are updated
            if math.fabs(opinion[A] - opinion[B]) <= threshold:
                oA = opinion[A]
                oB = opinion[B]
                opinion[A] = oA + beta * (oB -oA)
                opinion[B] = oB + beta * (oA - oB)
        # Plot the opinions of all individuals at the current time step on the scatter plot
        x = [i for _ in range (MAX_PERSON)]
        axe[1].scatter(x,opinion,c='red')
        axe[1].set_ylabel('Opinion')
    # Plot the opinion distribution on the histogram
    axe[0].hist(opinion,bins = 10,color='blue')
    axe[0].set_xlabel('Opinion')


def test_defuant():
    defuant(0.5,0.5)
    defuant(0.1,0.5)
    defuant(0.5,0.1)
    defuant(0.1,0.1)
    plt.show()   





'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''





if __name__ == "__main__":
    if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('-beta',default=0.3)
    parser.add_argument('-threshold',default=0.3)
    parser.add_argument('-test_default',action='store_true',default=False)
    parser.add_argument('-defuant',action='store_true',default = False)
    args = parser.parse_args()
    if args.test_defuant:
        test_defuant()
    else:
        defuant(float(args.beta),float(args.threshold))
