from networkx import Graph, from_edgelist, draw_networkx_nodes, draw_networkx_edges, draw_networkx_labels, spring_layout, set_node_attributes, set_edge_attributes, write_gexf, convert_node_labels_to_integers
from torch import manual_seed, cuda, matmul, argmax, randn, zeros
from matplotlib.pyplot import figure, show, title, savefig, box
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.nn import GCNConv, SAGEConv
from torch.nn import Dropout, Module, Embedding
from torch.nn.functional import relu, softmax
from torch.optim import AdamW
import numpy as np
import random

# Known chromatic numbers for specified problems (from references)
chromatic_numbers = {
    # COLOR graphs
    'jean.col': 10,
    'anna.col': 11,
    'myciel5.col': 6,
    'myciel6.col': 7,
    'queen5_5.col': 5,
    'queen6_6.col': 7,
    'queen7_7.col': 7,
    'queen8_8.col': 9,
    'queen9_9.col': 10,
    'queen8_12.col': 12,
    'queen11_11.col': 11,
}
#Color map for coloring nodes
color_map = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'yellow',
    4: 'orange',
    5: 'purple',
    6: 'pink',
    7: 'brown',
    8: 'grey',
    9: 'aqua',
    10: 'white',
    11: 'cyan',
    12: 'magenta',
    13: 'lime',
    14: 'teal',
    15: 'olive',
    16: 'maroon',
    17: 'navy',
    18: 'gold',
    19: 'indigo',
    20: 'tan',
}

#This is the ONLY function that I didn't write myself. It was taken from the Schuetz et al. repository even though it is not fundamental to the functioning of the models.
def set_seed(seed):
    """
    Sets random seeds for training.

    :param seed: Integer used for seed.
    :type seed: int
    """
    random.seed(seed) #Here we set the seed for the random module
    np.random.seed(seed) #Here we set the seed for the numpy module
    manual_seed(seed) #Here we set the seed for the torch module
    if cuda.is_available():
        cuda.manual_seed_all(seed) #Here we set the seed for the cuda module

def build_color_graph(file_name, file_path='./data/input/COLOR/instances/'):
    """
    Load problem definition (graph) from COLOR file (e.g. *.col).

    :param file_name: Filename of COLOR file
    :type fname: str
    :param file_path: Path to the file
    :type parent_fpath: str
    :return: Graph defined in provided file
    :rtype: networkx.Graph
    """

    path = file_path+file_name #Here we create the path to the file
    
    with open(path, 'r') as f: #Here we open the file
        content = f.read().strip() #Here we read the file and remove the spaces at the beginning and end of the file

    lines = [line for line in content.split('\n') if line.startswith('e')] #Here we create a list of the lines of the file that start with 'e' (this is where the edges are defined)
    edge_list = [tuple(map(int, line.split(' ')[1:])) for line in lines] #Here we create a list of tuples of the edges of the graph
    offset_edge_list = [tuple(map(lambda i, j: i - j, edge, (1,1))) for edge in edge_list] #Here we create a list of tuples of the edges of the graph with the offset of 1, so that the nodes start at 0

    unordered_graph = from_edgelist(offset_edge_list) #Here we create a graph from the list of edges

    G = Graph() #Here we create a new graph
    G.add_nodes_from(sorted(unordered_graph.nodes())) #Here we add the nodes to the graph
    G.add_edges_from(unordered_graph.edges) #Here we add the edges to the graph
    G = convert_node_labels_to_integers(G, first_label=0) #Here we convert the node labels to integers starting at 0
    #Here we return the graph
    return G

def get_edge_index(G, torch_device):
    """
    Get edge index and number of nodes from graph.

    :param G: Graph
    :type G: networkx.Graph
    :param dtype: Data type
    :type dtype: torch.dtype
    :return: Edge index
    :rtype: torch.Tensor
    """

    dataset = from_networkx(G) #Here we create the pytorch geometric dataset from the graph
    edge_index = dataset.edge_index #Here we get the edge index from the dataset
    #Here we return the edge index
    edge_index = edge_index.to(torch_device)
    return edge_index

def adjacency_matrix(edge_index, num_nodes, torch_dtype, torch_device):
    """
    Get adjacency matrix from edge index and number of nodes.

    :param edge_index: Edge index
    :type edge_index: torch.Tensor
    :param num_nodes: Number of nodes
    :type num_nodes: int
    :param torch_dtype: Data type
    :type torch_dtype: torch.dtype
    :param torch_device: Device
    :type torch_device: torch.device
    :return: Adjacency matrix
    :rtype: torch.Tensor
    """

    adj_matrix = zeros((num_nodes, num_nodes), dtype=torch_dtype) #Here we create a tensor of zeros with the shape of the adjacency matrix
    adj_matrix[edge_index[0], edge_index[1]] = 1 #Here we set the values of the adjacency matrix to 1 where there is an edge
    #Here we return the adjacency matrix
    adj_matrix = adj_matrix.to(torch_device)
    return adj_matrix

def plot_graph(G, node_colors, edge_colors, seed, node_size, figsize, name):
    """
    Plot graph.

    :param G: Graph
    :type G: networkx.Graph
    :param node_colors: Node colors
    :type node_colors: list
    :param edge_colors: Edge colors
    :type edge_colors: list
    :param seed: Seed
    :type seed: int
    :param node_size: Size of nodes
    :type node_size: int
    :param size: Size of figure
    :type size: int
    :param name: Name of figure
    :type name: str
    """

    figure(figsize = [figsize,figsize]) #Here we set the size of the figure

    draw_networkx_nodes(G,
        pos=spring_layout(G, seed=seed),
        node_size=node_size,
        node_color = node_colors
    ) #Here we draw the nodes of the graph

    draw_networkx_edges(G,
        pos=spring_layout(G, seed=seed),
        width=0.5,
        edge_color = edge_colors
    ) #Here we draw the edges of the graph

    draw_networkx_labels(G,
        pos=spring_layout(G, seed=seed),
        font_size=node_size/20,
        font_family="sans-serif",
        font_weight="bold"
    ) #Here we draw the labels of the graph

    title(name) #Here we set the title of the figure
    box(False) #Here we take off the box around the figure
    savefig('images/'+ name + '.png', bbox_inches='tight') #Here we save the figure
    show() #Here we show the figure

class GCN(Module):
    """
    GCN-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters.
    """
    def __init__(self, in_features, hidden_size, num_colors, dropout, seed):
        """
        Initialize the model object. Establishes model architecture and relevant hyperparameters.

        :param in_features: Number of input features
        :type in_features: int
        :param hidden_size: Number of hidden units
        :type hidden_size: int
        :param num_colors: Number of colors
        :type num_colors: int
        :param dropout: Dropout probability
        :type dropout: float
        :param seed: Seed
        :type seed: int
        """
        super(GCN, self).__init__() #Here we initialize the model object

        manual_seed(seed) #Here we set the seed
        self.conv1 = GCNConv(in_features, hidden_size) #Here we create the first convolutional layer
        self.conv2 = GCNConv(hidden_size, num_colors) #Here we create the second convolutional layer
        self.dropout = Dropout(dropout) #Here we create the dropout layer

    def forward(self, x, edge_index):
        """
        Define forward step of network. In this example, pass inputs through convolution, apply relu
        and dropout, then pass through second convolution. Finally apply softmax to get class logits.

        :param x: Input features
        :type x: torch.Tensor
        :param edge_index: Edge index
        :type edge_index: torch.Tensor
        :return: Class logits
        :rtype: torch.Tensor
        """
        x = self.conv1(x, edge_index) #Here we pass the input through the first convolutional layer
        x = relu(x) #Here we apply the relu activation function
        x = self.dropout(x) #Here we apply the dropout layer
        x = self.conv2(x, edge_index) #Here we pass the input through the second convolutional layer
        #Here we apply the softmax activation function to get each color's probability
        return softmax(x, dim=1) 
    
class SAGE(Module):
    """
    GraphSAGE-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters.
    """
    def __init__(self, in_features, hidden_size, num_colors, dropout, seed):
        """
        Initialize the model object. Establishes model architecture and relevant hyperparameters.

        :param in_features: Number of input features
        :type in_features: int
        :param hidden_size: Number of hidden units
        :type hidden_size: int
        :param num_colors: Number of colors
        :type num_colors: int
        :param dropout: Dropout probability
        :type dropout: float
        :param seed: Seed
        :type seed: int
        """
        super(SAGE, self).__init__() #Here we initialize the model object
        
        manual_seed(seed) #Here we set the seed
        self.conv1 = SAGEConv(in_features, hidden_size, aggr='mean') #Here we create the first convolutional layer
        self.conv2 = SAGEConv(hidden_size, num_colors, aggr='mean') #Here we create the second convolutional layer
        self.dropout = Dropout(p=dropout) #Here we create the dropout layer
        

    def forward(self, x, edge_index):
        """
        Define forward step of network. In this example, pass inputs through convolution, 
        apply relu and dropout, then pass through second convolution. Finally apply softmax to get class logits.

        :param x: Input features
        :type x: torch.Tensor
        :param edge_index: Edge index
        :type edge_index: torch.Tensor
        :return: Class logits
        :rtype: torch.Tensor
        """
        x = self.conv1(x, edge_index) #Here we pass the input through the first convolutional layer
        x = relu(x) #Here we apply the relu activation function
        x = self.dropout(x) #Here we apply the dropout layer
        x = self.conv2(x, edge_index) #Here we pass the input through the second convolutional layer
        #Here we apply the softmax activation function to get each color's probability
        return softmax(x, dim=1) 
    
def potts_hard_loss(G, coloring):
    """
    Function to calculate Potts Model (hard) loss of graph coloring.

    :param G: Graph
    :type G: networkx.Graph
    :param coloring: Coloring
    :type coloring: torch.tensor
    :return: Cost of provided class assignments
    :rtype: float
    """
    cost = 0.
    for edge in G.edges(): #Here we iterate through the edges of the graph
        if coloring[edge[0]] == coloring[edge[1]]: #Here we check if the nodes of the edge have the same color
            cost += 1. #Here we increment the cost by 1 if the nodes have the same color
    #Here we return the cost
    return cost

def potts_soft_loss(probs, adjacency_matrix):
    """
    Function to calculate Potts Model (soft) loss based on soft assignments (probabilities)

    :param probs: Probability vector, of each node belonging to each class
    :type probs: torch.tensor
    :param adjacency_matrix: Adjacency matrix
    :type adjacency_matrix: torch.tensor
    :return: Loss, given the current soft assignments (probabilities)
    :rtype: torch.tensor
    """
    loss = (adjacency_matrix * matmul(probs, probs.T)).sum() / 2. #Here we calculate the loss by multiplying the adjacency matrix by the outer product of the probability vector with itself, and then summing over all entries. We divide by 2 to avoid double counting.
    #Here we return the loss
    return loss

def initial_embedding(number_nodes, dim_embedding, torch_device, torch_dtype):
    """
    Function to create initial embedding.

    :param number_nodes: Number of nodes
    :type number_nodes: int
    :param dim_embedding: Dimension of embedding
    :type dim_embedding: int
    :param torch_device: Torch device
    :type torch_device: torch.device
    :param torch_dtype: Torch dtype
    :type torch_dtype: torch.dtype
    :return: Initial embedding
    :rtype: torch.tensor
    """
    #Here we create the initial embedding by sampling from a normal distribution with mean 0 and variance 1
    init_embedding = Embedding(num_embeddings = number_nodes, embedding_dim=dim_embedding, device=torch_device, dtype=torch_dtype).weight
    return init_embedding


def initialize_model(hyperparameters):
    """
    Helper function to load in the GNN model, ADAM optimizer and initial embedding.

    :param hyperparameters: Hyperparameters
    :type hyperparameters: dict
    :return: Model, optimizer, initial embedding
    :rtype: torch.nn.Module, torch.optim.Optimizer
    """
    seed = hyperparameters["seed"] #Here we get the seed
    print(f'Setting seed to {seed}...') #Here we print the seed
    set_seed(seed) #Here we set the seed

    model = hyperparameters['model'] #Here we get the model
    initial_dim = hyperparameters['initial_dim'] #Here we get the embedding dimension
    hidden_dim = hyperparameters['hidden_dim'] #Here we get the hidden dimension
    dropout = hyperparameters['dropout'] #Here we get the dropout probability
    number_colors = hyperparameters['number_colors'] #Here we get the number of colors
    learning_rate = hyperparameters['learning_rate'] #Here we get the learning rate
    torch_device = hyperparameters['device'] #Here we get the torch device
    torch_dtype = hyperparameters['dtype'] #Here we get the torch dtype
    num_nodes = hyperparameters['num_nodes'] #Here we get the number of nodes

    # Here we initialize the model
    if model == "GCN":
        print(f'Building {model} model...') 
        net = GCN(initial_dim, hidden_dim, number_colors, dropout, seed)  #Here we create the GCN model
    elif model == "GraphSAGE":
        print(f'Building {model} model...') #Here we print the model
        net = SAGE(initial_dim, hidden_dim, number_colors, dropout, seed) #Here we create the GraphSAGE model
    else:
        raise ValueError("Model only supports 'GCN' or 'GraphSAGE'") #Here we raise an error if the model type is invalid

    net = net.type(torch_dtype).to(torch_device) #Here we set the model to the device and dtype
    embedding = initial_embedding(num_nodes, initial_dim, torch_device, torch_dtype) #Here we initialize the embedding
    
    # Here we initialize the optimizer
    print('Building ADAM-W optimizer...') #Here we print the optimizer
    optimizer = AdamW(net.parameters(), lr= learning_rate, weight_decay=1e-2) #Here we create the optimizer

    #Here we return the model, optimizer, embedding, edge index, and adjacency matrix
    return net, embedding, optimizer


def training(G, net, initial_embedding, optimizer, hyperparameters, verbose=True):
    """
    Function to train the GNN model.

    :param G: Graph
    :type G: networkx.Graph
    :param net: GNN model
    :type net: torch.nn.Module
    :param initial_embedding: Initial embedding
    :type initial_embedding: torch.tensor
    :param optimizer: Optimizer
    :type optimizer: torch.optim.Optimizer
    :param hyperparameters: Hyperparameters
    :type hyperparameters: dict
    :param verbose: Whether to print training progress, defaults to True
    :type verbose: bool, optional
    :return: probability vector, best coloring, best loss, best hard loss
    :rtype: torch.tensor, torch.tensor, torch.tensor, float
    """
    
    seed = hyperparameters["seed"] #Here we get the seed
    print(f'Setting seed to {seed}...') #Here we print the seed
    set_seed(seed) #Here we set the seed

    number_epochs = hyperparameters['number_epochs'] #Here we get the number of epochs
    torch_device = hyperparameters['device'] #Here we get the torch device
    torch_dtype = hyperparameters['dtype'] #Here we get the torch dtype
    patience = hyperparameters['patience'] #Here we get the patience
    num_nodes = hyperparameters['num_nodes'] #Here we get the number of nodes

    edge_index = get_edge_index(G, torch_device) #Here we get the edge index
    adj_mat = adjacency_matrix(edge_index, num_nodes, torch_dtype, torch_device) #Here we get the adjacency matrix


    prev_loss = np.inf  # We initialize the previous loss as a big number
    best_loss = np.inf  # We initialize the best loss as a big number
    counter = 0  # We initialize the patience counter


    # This is the training loop
    for epoch in range(number_epochs):
        net.train() #Here we set the model to train mode
        probs = net(initial_embedding, edge_index) #Here we get the probabilities from the model
        loss = potts_soft_loss(probs, adj_mat) #Here we calculate the soft loss of the probabilities
        
        coloring = argmax(probs, dim=1) #Here we get the coloring by taking the argmax of the probabilities
        loss_hard = potts_hard_loss(G, coloring) #Here we calculate the hard loss given by the coloring
        
        # If loss increases or change in loss is too small, we increase the patience counter
        if ((loss > prev_loss) or (abs(loss - prev_loss) < 1e-4)):
            counter += 1
            if counter >= patience: #If patience is reached, we stop the training
                print(f'Stopping early on epoch {epoch}!')
                break
        else: #If loss decreases, we reset the patience counter
            counter = 0
            if loss < best_loss: #If loss is less than the best loss, we update the best loss, best coloring and best hard loss
                best_loss = loss
                best_hard_loss = loss_hard
                best_coloring = coloring

        prev_loss = loss #Here we update the previous loss

        optimizer.zero_grad()  # Here we zero the gradients
        loss.backward()  # Here we backpropagate
        optimizer.step()  # Here we update the weights

        if (verbose == True) & ( epoch % 1000 == 0): #Here we check if we should print the results
            print(f'Epoch: {epoch} | Soft Loss: {loss.item()} | Hard Loss: {loss_hard} | Patience: {counter}')

    print(f'Best Soft Loss: {best_loss.item()}, Best Hard Loss: {best_hard_loss}, Best Coloring: {best_coloring}') #Here we print the final coloring and loss
    
    # Here we return the probabilities, the best coloring, the best soft loss and the best hard loss
    return probs, best_coloring, best_loss, best_hard_loss


def new_coloring(G, coloring):
    """
    Function that takes the edges of a graph that contain the same color,
    randomly chooses one node of each edge and gives it a new color that minimizes
    the cost function.
    :param G: Graph instance to solve
    :type G: networkx.Graph
    :param coloring: Previous coloring
    :type coloring: torch.tensor
    :return: New coloring, number of colors used
    :rtype: torch.tensor, int
    """
    loss = potts_hard_loss(G, coloring) #Here we calculate the loss of the coloring
    while loss != 0.0: #Here we check if the loss is 0
        max_color = max(coloring) #Here we get the maximum color
        for edge in G.edges: 
            if coloring[edge[0]] == coloring[edge[1]]: #Here we check if the nodes of the edge have the same color
                coloring[edge[random.randint(0,1)]] = max_color + 1 #Here we give a new color to one of the nodes that have the same color
        loss = potts_hard_loss(G, coloring) #Here we calculate the loss of the coloring again

    #Here we return the new coloring and the number of colors used
    return coloring, max(coloring)+1 

def color_lists(G, coloring, color_map):
    """
    Function that gives the lists of node colors and edge colors of a graph
    :param G: Graph instance to solve
    :type G: networkx.Graph
    :param coloring: Best coloring found by the model
    :type coloring: torch.tensor
    :param color_map: Dictionary of colors
    :type color_map: dict
    :return: Lists of node colors and edge colors
    :rtype: list, list
    """

    coloring_list = coloring.tolist() #Here we convert the coloring to a list
    node_colors= [color_map[i] for i in coloring_list] #Here we get the list of node colors
    
    #Here we get the list of edge colors. If the nodes of the edge have the same color, the edge color is red, otherwise it is black
    edge_colors = ['red' if coloring[edge[0]] == coloring[edge[1]] else 'black' for edge in G.edges]

    return node_colors, edge_colors

def save_graph(G, node_colors, edge_colors, name, path = 'graphs/'):
    """
    Function that saves a graph
    :param G: Graph instance to solve
    :type G: networkx.Graph
    :param node_colors: List of node colors
    :type node_colors: list
    :param edge_colors: List of edge colors
    :type edge_colors: list
    :param path: Path to save the graph
    :type path: str
    :param name: Name of the graph
    """
    node_color_dict = {}
    for i in range(len(node_colors)):
        node_color_dict[i] = node_colors[i]
    
    edge_color_dict = {}
    edges = list(G.edges)
    for i in range(len(edges)):
        edge_color_dict[edges[i]] = edge_colors[i]

    set_node_attributes(G, node_color_dict, 'node_color')
    set_edge_attributes(G, edge_color_dict, 'edge_color')

    write_gexf(G, path + name + '.gexf')
