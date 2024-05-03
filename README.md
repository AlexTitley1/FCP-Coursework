This is my submission for the FCP assignment.
(https://github.com/AlexTitley1/FCP-Coursework)

Task 1
This script implements the Ising model, which simulates a two-dimensional spin grid that can interact with its nearest neighbor and respond to an external magnetic field. The simulation examines how the spins are aligned or relative under various conditions, affected by temperature and magnetic field strength. The program is initialised with a 100*100 grid consisting of -1 and 1 randomly, simulating the initial state of spin up and down. The external magnetic field strength and temperature inverse (which is the parameter of alpha) are controlled by command line parameters. Then different key functions (calculate_agreement, ising_step, plot_ising and test_ising) are used for debugging and optimisation, and for each cell in the grid, interactions with its neighbours are calculated, and then based on the results and the value of alpha, a decision is made whether to invert the spin or not, and finally the simulation is run to obtain a visualisation of the Results.

Task 2
This script implements the Deffuant model of opinion dynamics, where each individual (or "agent") within a closed group updates their opinion through pairwise interactions. The model allows interactions only if the opinions of the agents are within a certain threshold of each other. The strength of the opinion shift during interactions is governed by a coupling constant, beta. To run the simulation, execute the script from the command line. You can test different parameters by modifying the test_defuant function within the script.
The script generates two plots:
A scatter plot showing the evolution of opinions for each time step.
A histogram showing the final distribution of opinions.

Task 3
This script creates a random network of a chosen number of nodes which the user specifies. It calculates the mean dgeree, mean path length, mean clustering coefficient and also a plot of this random network. To run task three, use the flag "-network N", where N is an integer of chosing to represent the number of nodes. To test the network run the flag "-test_network".

Task 4
This script implements both ring networks and small world networks. The ring networks take 2 arguments, amount of nodes and neighbour range. The amount of nodes is how many nodes the network will consist of and each node will have an index incrementing by one until the number of nodes has been reached. The neighbour range is related to the ring nature of the graph. It is used to see how many neighbour nodes each node should connect to. This is done by connecting the node to every node withing +/- the neighbour range. The small worlds network uses a ring network with a neighbour range of 2 and rewires some of the connections to more accurately model the connections between people in the real world. The small world network takes in 2 arguments, the number of nodes and the rewire probability. The number of nodes is used to generate a ring network with the same number of nodes and a neighbour range of 2. The rewire probability is used by generrating a random number and checking whether it is above or below the probability. A list of possible connections in the network is created and for eac possible rewire the program generates a random number and if it is below the rewire probability, a list of possible rewire destinations is created and one is selected at random to be the new place the edge connects.

Task 5
