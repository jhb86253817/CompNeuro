# This is the template code for Programming Project 1 - 
# Option 1: Backpropagation and Autoencoders.
#
# You can add command-line arguments if you'd like (see below __main__).
#
# You can add 'parameters' to the functions in the respective arguments.
# They are called 'initialization_params', 'training_params', and 'test_params'.
# Think of these as means of controlling how your code works under different
# settings. They are supposed to be python lists, so you can use them eg.:
#    learning_rate = training_params[0]
#    stopping_criterion = training_params[1]
# etc.
# You may also not use them if you do not need them, eg. if initialization
# is completely random, initialization_params may not be necessary.
#
# XXX_state (eg. "feedforward_classifier_state") is a variable that is supposed to hold
# current firing status of all the nodes in the network.
# XXX_connections (eg. "feedforward_classifier_connections") is a variable that is supposed
# to hold the weights of the connections between the nodes.
# You have flexibility designing these variables.
# eg, you may have a flat vector of dimensionality N for XXX_states, where N is the total number
# of the nodes in the network, or you can as well have a list of state vectors, where each 
# element (vector) in the list corresponds to one of the layers.
# You can as well design XXX_connections variables as you wish, ie, use a flattened or 
# layer-by-layer representation, or anything you would like really.
#
# You do not necessarily have to use every argument in every function, 
# ie. feedforward_classifier_state argument is probably redundant in test_feedforward_classifier().
#
# You are very welcome to add more functions: They can be auxilary 
# functions, plotting functions, anything you wish.
#
# Also feel free to add global variables if you like.
#
# In __main__ as well, you can add any additional code anywhere.
# eg., you may consider having a loop for varying parameters.
# You may also change the "flow" of the code, eg, you can collect
# test data after training is complete: You have flexibility with
# this code part really.
#
# For your "matrices":
# You can use either the array or matrix type of numpy, or even switch 
# between them as you wish.
#
# Enjoy! :)
# Hande
 
 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg,stats
import pprint
# Uncomment if you will use the GUI code:
# import gui_template as gt


# Initialize the corresponding networks
def init_feedforward_classifier(initialization_params):
    # Place your code here
    feedforward_classifier_state = [] 
    for num in initialization_params:
        feedforward_classifier_state.append(np.zeros((num, 1)))
    feedforward_classifier_connections = []
    for i in range(len(initialization_params)-1):
        num_row = initialization_params[i] + 1
        num_col = initialization_params[i+1]
        feedforward_classifier_connections.append(np.random.rand(num_row, num_col))
    return [feedforward_classifier_state, feedforward_classifier_connections]

def init_autoencoder(initialization_params):
    # Place your code here
    return [autoencoder_state, autoencoder_connections]

def init_autoencoder_classifier(initialization_params):
    # Place your code here
    return [autoencoder_classifier_state, autoencoder_classifier_connections]
    
    
    
def sigmoid(o):
    return 1.0 / (1 + np.exp(-o))

# Given an input, these functions calculate the corresponding output to 
# that input.
def update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections):
    # Place your code here
    for i in range(len(feedforward_classifier_state)-1):
        temp = np.array([np.insert(feedforward_classifier_state[i], len(feedforward_classifier_state[i]), 1)])
        feedforward_classifier_state[i+1] = (temp.dot(feedforward_classifier_connections[i])).T
        feedforward_classifier_state[i+1] = sigmoid(feedforward_classifier_state[i+1])

    return feedforward_classifier_state
    
def update_autoencoder(autoencoder_state, autoencoder_connections):
    # Place your code here
    return autoencoder_state
    
def update_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections):
    # Place your code here
    return autoencoder_classifier_state
    
# Backpropagation
def bp_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, target, training_params):
    connections_num = len(feedforward_classifier_connections)
    gradients = []
    deltas = []
    for i in range(connections_num):
        o = feedforward_classifier_state[connections_num-i]
        D = np.diag((o*(1-o))[:,0])
        if i == 0:
            e = o - target
            delta = D.dot(e)
        else:
            delta = D.dot(feedforward_classifier_connections[connections_num-i][:-1,:].dot(deltas[-1]))
        deltas.append(delta)
        o_pre = feedforward_classifier_state[connections_num-i-1]
        gradient = - training_params[0] * (delta.dot(np.array([np.insert(o_pre, len(o_pre), 1)]))).T
        gradients.append(gradient)
    return gradients[::-1]

def bp_autoencoder():
    pass

def bp_autoencoder_classifier():
    pass

def cost_feedforward_classifier(feedforward_classifier_state, target):
    o = feedforward_classifier_state[-1]
    cost = np.sum(np.square(o-target)) / 2.0
    return cost
        
        
# Main functions to handle the training of the networks. 
# Feel free to write auxiliary functions and call them from here.
# These functions are supposed to call the update functions.
def train_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, training_data, training_params):
    # Place your code here
    data_num = len(training_data[0])
    costs = []
    for i in range(900):
        cost = 0.0
        gradients_cml = None
        #gradients_cml.append(np.zeros((3,2)))
        #gradients_cml.append(np.zeros((3,1)))
        for j in range(data_num):
            o = training_data[0][j].T
            target = training_data[1][j].T
            feedforward_classifier_state[0] = o
            feedforward_classifier_state = update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections)
            cost += cost_feedforward_classifier(feedforward_classifier_state, target)
            gradients = bp_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, target, training_params)
            if j==0:
                gradients_cml = gradients
            else:
                for k in range(len(gradients)):
                    gradients_cml[k] += gradients[k]
        for k in range(len(gradients_cml)):
            feedforward_classifier_connections[k] += gradients_cml[k]

        costs.append(cost)
    ################################################################### 
    plt.plot(range(1,901), costs)
    plt.show()
    # Please do output your training performance here
    return feedforward_classifier_connections
    
def train_autoencoder(autoencoder_state, autoencoder_connections, training_data, training_params):
    # Place your code here
    
    # Please do output your training performance here
    return autoencoder_connections

def train_autoencoder(autoencoder_classifier_state, autoencoder_classifier_connections, training_data, training_params):
    # Place your code here
    
    # Please do output your training performance here
    return autoencoder_classifier_connections



# Main functions to handle the testing of the networks. 
# Feel free to write auxiliary functions and call them from here.
# These functions are supposed to call the 'run' functions.
def test_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, test_data, test_params):
    # Place your code here
    
    # Please do output your test performance here
    None
    
def test_autoencoder(autoencoder_state, autoencoder_connections, test_data, test_params):
    # Place your code here
    
    # Please do output your test performance here
    None

def test_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, test_data, test_params):
    # Place your code here
    
    # Please do output your test performance here
    None
    


# You may also want to be able to save and load your networks, which will 
# help you for debugging weird behavior. (Consider this seriously if you
# are having debugging problems.)

# def save_feedforward_classifier(filename, feedforward_classifier_state, feedforward_classifier_connections):
#     None
# def load_feedforward_classifier(filename):
#     return [feedforward_classifier_state, feedforward_classifier_connections]

# def save_autoencoder(filename, autoencoder_state, autoencoder_connections):
#     None
# def load_autoencoder(filename):
#     return [autoencoder_state, autoencoder_connections]

# def save_autoencoder_classifier(filename, autoencoder_classifier_state, autoencoder_classifier_connections):
#     None
# def load_autoencoder(filename):
#     return [autoencoder_classifier_state, autoencoder_classifier_connections]



if __name__=='__main__':

    # Please use the following snippet for clarity if you have command-line 
    # arguments:

    # if len(argv) < <number_of_expected_arguments>:
    #    print('Usage: python', argv[0], '<expected_arg_1>', '<expected_arg_2>, ...')
    #    exit(1)
    
    # arg_1 = argv[1]
    # arg_2 = argv[2]
    # ...
    
    
    # Read data here
    training_data = None 
    # use XOR function to test
    training_data = []
    training_data.append([])
    training_data[0].append(np.array([[0, 0]]))
    training_data[0].append(np.array([[0, 1]]))
    training_data[0].append(np.array([[1, 0]]))
    training_data[0].append(np.array([[1, 1]]))
    training_data.append([])
    training_data[1].append(np.array([[0]]))
    training_data[1].append(np.array([[1]]))
    training_data[1].append(np.array([[1]]))
    training_data[1].append(np.array([[0]]))
    #print training_data[0]
    #print training_data[1]

    test_data = None
    
    # You may also use the gui_template.py functions to collect image data from the user. eg:
    # training_data = gt.get_images()
    # if len(training_data) != 0:
	#     gt.visualize_image(training_data[-1])
    
    # If you wish, you can have a loop here, or any other place really, that will update the
    # parameters below automatically.
    
    # Initialize network(s) here
    # each number stands for the number of neurons in each layer
    # from left to right are input layer, hidden layer(s), output layer
    initialization_params = [2, 2, 1]       
    feedforward_classifier_state = None
    feedforward_classifier_connections = None 
    [feedforward_classifier_state, feedforward_classifier_connections] = init_feedforward_classifier(initialization_params)
    # Change initialization params if desired
    autoencoder_state = None
    autoencoder_connections = None
    [autoencoder_state, autoencoder_connections] = init_autoencoder(initialization_params)
    # Change initialization params if desired
    autoencoder_classifier_state = None
    autoencoder_classifier_connections = None
    [autoencoder_classifier_state, autoencoder_classifier_connections] = init_autoencoder_classifier(initialization_params)
    
    
    # Train network(s) here
    training_params = [1]
    feedforward_classifier_connections = train_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, training_data, training_params)
    # Change training params if desired
    autoencoder_connections = train_autoencoder(autoencoder_state, autoencoder_connections, training_data, training_params)
    # Change training params if desired
    autoencoder_classifier_connections = train_autoencoder(autoencoder_classifier_state, autoencoder_classifier_connections, training_data, training_params)
   
    
    # Test network(s) here
    test_params = None
    test_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, test_data, test_params)
    # Change test params if desired
    test_autoencoder(autoencoder_state, autoencoder_connections, test_data, test_params)
    # Change test params if desired
    test_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, test_data, test_params)
	
    # You can use gui_template.py functions for visualization as you wish
