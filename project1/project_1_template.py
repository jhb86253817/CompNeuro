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
 

from __future__ import division 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg,stats
import cPickle, gzip
import json
# Uncomment if you will use the GUI code:
# import gui_template as gt

# fix random seed
np.random.seed(123)

# Initialize the corresponding networks
def init_feedforward_classifier(initialization_params):
    # Place your code here
    layers_size = initialization_params[0]
    data_num = initialization_params[1]

    feedforward_classifier_state = [] 
    for layer_size in layers_size:
        feedforward_classifier_state.append(np.zeros((data_num, layer_size)))
    feedforward_classifier_connections = []
    for i in range(len(layers_size)-1):
        num_row = layers_size[i] + 1
        num_col = layers_size[i+1]
        # random initialization with interval [-0.1, 0.1]
        feedforward_classifier_connections.append((np.random.rand(num_row, num_col) - 0.5) / 5)
    return [feedforward_classifier_state, feedforward_classifier_connections]

def init_autoencoder(initialization_params):
    # Place your code here
    layers_size = initialization_params[0]
    data_num = initialization_params[1]

    autoencoder_state = [] 
    for layer_size in layers_size:
        autoencoder_state.append(np.zeros((data_num, layer_size)))
    autoencoder_connections = []
    for i in range(len(layers_size)-1):
        num_row = layers_size[i] + 1
        num_col = layers_size[i+1]
        # random initialization with interval [-0.1, 0.1]
        autoencoder_connections.append((np.random.rand(num_row, num_col) - 0.5) / 5)
    return [autoencoder_state, autoencoder_connections]

def init_autoencoder_classifier(initialization_params):
    # Place your code here
    layers_size = initialization_params[0]
    data_num = initialization_params[1]
    first_layer_connection = initialization_params[2]

    autoencoder_classifier_state = [] 
    for layer_size in layers_size:
        autoencoder_classifier_state.append(np.zeros((data_num, layer_size)))
    autoencoder_classifier_connections = []
    autoencoder_classifier_connections.append(first_layer_connection)
    for i in range(1, len(layers_size)-1):
        num_row = layers_size[i] + 1
        num_col = layers_size[i+1]
        # random initialization with interval [-0.1, 0.1]
        autoencoder_classifier_connections.append((np.random.rand(num_row, num_col) - 0.5) / 5)
    return [autoencoder_classifier_state, autoencoder_classifier_connections]
    
def sigmoid(o):
    return 1.0 / (1 + np.exp(-o))

# Given an input, these functions calculate the corresponding output to 
# that input.
def update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections):
    # Place your code here
    data_num = feedforward_classifier_state[0].shape[0]
    bias = np.ones((data_num, 1))
    for i in range(len(feedforward_classifier_state)-1):
        state_temp = np.concatenate((feedforward_classifier_state[i], bias), axis = 1)
        feedforward_classifier_state[i+1] = state_temp.dot(feedforward_classifier_connections[i])
        feedforward_classifier_state[i+1] = sigmoid(feedforward_classifier_state[i+1])
    return feedforward_classifier_state
    
def update_autoencoder(autoencoder_state, autoencoder_connections):
    # Place your code here
    data_num = autoencoder_state[0].shape[0]
    bias = np.ones((data_num, 1))
    for i in range(len(autoencoder_state)-1):
        state_temp = np.concatenate((autoencoder_state[i], bias), axis = 1)
        autoencoder_state[i+1] = state_temp.dot(autoencoder_connections[i])
        autoencoder_state[i+1] = sigmoid(autoencoder_state[i+1])
    return autoencoder_state
    
def update_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections):
    # Place your code here
    data_num = autoencoder_classifier_state[0].shape[0]
    bias = np.ones((data_num, 1))
    for i in range(len(autoencoder_classifier_state)-1):
        state_temp = np.concatenate((autoencoder_classifier_state[i], bias), axis = 1)
        autoencoder_classifier_state[i+1] = state_temp.dot(autoencoder_classifier_connections[i])
        autoencoder_classifier_state[i+1] = sigmoid(autoencoder_classifier_state[i+1])
    return autoencoder_classifier_state
    
# Backpropagation
def bp_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, target, training_params):
    data_num = feedforward_classifier_state[0].shape[0]
    connections_num = len(feedforward_classifier_connections)
    bias = np.ones((data_num, 1))
    gradients = []
    deltas = []
    for i in range(connections_num):
        output = feedforward_classifier_state[connections_num-i]
        if i == 0:
            error = output - target
            delta = output * (1 - output) * error
        else:
            delta = output * (1 - output) * deltas[-1].dot(feedforward_classifier_connections[connections_num-i][:-1,:].T)
        deltas.append(delta)
        output_pre = feedforward_classifier_state[connections_num-i-1]
        output_pre = np.concatenate((output_pre, bias), axis = 1)
        gradient = - training_params[0] * output_pre.T.dot(delta)
        gradients.append(gradient)
    return gradients[::-1]

def bp_autoencoder(autoencoder_classifier_state, autoencoder_classifier_connections, target, training_params):
    data_num = autoencoder_classifier_state[0].shape[0]
    connections_num = len(autoencoder_classifier_connections)
    rho = training_params[2]
    beta = training_params[3]
    rho_hat = np.mean(autoencoder_state[1], axis=0)
    bias = np.ones((data_num, 1))
    gradients = []
    deltas = []
    for i in range(connections_num):
        output = autoencoder_state[connections_num-i]
        if i == 0:
            error = output - target
            delta = output * (1 - output) * error
        else:
            sparse_penalty = beta * (-1.0 * rho / rho_hat + (1.0 - rho) / (1.0 - rho_hat))
            delta =  sparse_penalty + deltas[-1].dot(autoencoder_connections[connections_num-i][:-1,:].T)
            delta = delta * output * (1.0 - output)
        deltas.append(delta)
        output_pre = autoencoder_state[connections_num-i-1]
        output_pre = np.concatenate((output_pre, bias), axis = 1)
        gradient = - training_params[0] * output_pre.T.dot(delta)
        gradients.append(gradient)
    return gradients[::-1]


def bp_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, target, training_params):
    data_num = autoencoder_classifier_state[0].shape[0]
    connections_num = len(autoencoder_classifier_connections)
    bias = np.ones((data_num, 1))
    gradients = []
    deltas = []
    for i in range(connections_num - 1):
        output = autoencoder_classifier_state[connections_num-i]
        if i == 0:
            error = output - target
            delta = output * (1.0 - output) * error
        else:
            delta = output * (1.0 - output) * deltas[-1].dot(autoencoder_classifier_connections[connections_num-i][:-1,:].T)
        deltas.append(delta)
        output_pre = autoencoder_classifier_state[connections_num-i-1]
        output_pre = np.concatenate((output_pre, bias), axis = 1)
        gradient = - training_params[0] * output_pre.T.dot(delta)
        gradients.append(gradient)
    return gradients[::-1]

def cost_feedforward_classifier(output, target):
    cost = np.sum(np.square(output - target)) / 2
    return cost

def cost_autoencoder(autoencoder_state, target, training_params):
    output = autoencoder_state[-1]
    rho_hat = np.mean(autoencoder_state[1], axis=0)
    rho = training_params[2]
    cost1 = np.sum(np.square(output - target)) / 2
    cost2 = rho * np.log(1.0 * rho / rho_hat) + (1.0 - rho) * np.log((1.0-rho) / (1.0 - rho_hat))
    cost2 = np.sum(cost2)
    return cost1 + cost2
        
def cost_autoencoder_classifier(output, target):
    cost = np.sum(np.square(output - target)) / 2
    return cost

# Main functions to handle the training of the networks. 
# Feel free to write auxiliary functions and call them from here.
# These functions are supposed to call the update functions.
def train_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, training_data, test_data, training_params):
    print "start training feedforward classifier..."
    # Place your code here
    data_num = training_data[0].shape[0]
    iteration_num = training_params[1]
    costs = []
    for i in range(iteration_num):
        accuracy = test_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, test_data)
        feedforward_classifier_state[0] = training_data[0]
        feedforward_classifier_state = update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections)
        cost = cost_feedforward_classifier(feedforward_classifier_state[-1], training_data[1])
        costs.append(cost)
        print "iteration %d, cost: %f, accuracy: %d" % (i+1, cost, accuracy)
        # compute gradients using backpropagation
        gradients = bp_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, training_data[1], training_params)
        # update weights using the gradients
        for k in range(len(gradients)):
            feedforward_classifier_connections[k] += gradients[k]

    # Please do output your training performance here
    return feedforward_classifier_connections
    
def train_autoencoder(autoencoder_state, autoencoder_connections, training_data, test_data, training_params):
    print "start training autoencoder..."
    # Place your code here
    data_num = training_data[0].shape[0]
    iteration_num = training_params[1]
    costs = []
    for i in range(iteration_num):
        cost_test = test_autoencoder(autoencoder_state, autoencoder_connections, test_data, training_params)
        autoencoder_state[0] = training_data[0]
        autoencoder_state = update_autoencoder(autoencoder_state, autoencoder_connections)
        cost = cost_autoencoder(autoencoder_state, training_data[0], training_params)
        costs.append(cost)
        print "iteration %d, cost: %f, cost_test: %f" % (i+1, cost, cost_test)
        # compute gradients using backpropagation
        gradients = bp_autoencoder(autoencoder_state, autoencoder_connections, training_data[0], training_params)
        # update weights using the gradients
        for k in range(len(gradients)):
            autoencoder_connections[k] += gradients[k]
    
    # Please do output your training performance here
    return autoencoder_connections

def train_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, training_data, test_data, training_params):
    print "start training autoencoder classifier..."
    # Place your code here
    data_num = training_data[0].shape[0]
    iteration_num = training_params[1]
    costs = []
    for i in range(iteration_num):
        accuracy = test_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, test_data)
        autoencoder_classifier_state[0] = training_data[0]
        autoencoder_classifier_state = update_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections)
        cost = cost_autoencoder_classifier(autoencoder_classifier_state[-1], training_data[1])
        costs.append(cost)
        print "iteration %d, cost: %f, accuracy: %d" % (i+1, cost, accuracy)
        # compute gradients using backpropagation
        gradients = bp_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, training_data[1], training_params)
        # update weights using the gradients
        for k in range(len(gradients)):
            autoencoder_classifier_connections[k+1] += gradients[k]
    
    # Please do output your training performance here
    return autoencoder_classifier_connections


# Main functions to handle the testing of the networks. 
# Feel free to write auxiliary functions and call them from here.
# These functions are supposed to call the 'run' functions.
def test_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, test_data):
    # Place your code here
    feedforward_classifier_state[0] = test_data[0]
    feedforward_classifier_state = update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections)
    # do prediction as the class with the largest output
    output = np.argmax(feedforward_classifier_state[-1], axis = 1)
    # labels of the test data
    target = test_data[1]
    correct = list(output - target).count(0)
    
    # Please do output your test performance here
    #return 1.0 * correct / len(test_data[1])
    return correct

    #feedforward_classifier_state[0] = test_data[0]
    #feedforward_classifier_state = update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections)
    #accuracy = cost_feedforward_classifier(feedforward_classifier_state[-1], test_data[1])
    #return accuracy
    
def test_autoencoder(autoencoder_state, autoencoder_connections, test_data, training_params):
    # Place your code here
    autoencoder_state[0] = test_data[0]
    autoencoder_state = update_autoencoder(autoencoder_state, autoencoder_connections)
    output = autoencoder_state[-1]
    target = test_data[0]
    cost_test = cost_autoencoder(autoencoder_state, target, training_params)
    
    # Please do output your test performance here
    return cost_test

def test_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, test_data):
    # Place your code here
    autoencoder_classifier_state[0] = test_data[0]
    autoencoder_classifier_state = update_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections)
    # do prediction as the class with the largest output
    output = np.argmax(autoencoder_classifier_state[-1], axis = 1)
    # labels of the test data
    target = test_data[1]
    correct = list(output - target).count(0)
    
    # Please do output your test performance here
    #return 1.0 * correct / len(test_data[1])
    return correct


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
    f = gzip.open('MNIST_theano/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # treat both training data and validate data as training data
    # 2d array, 60000 x 784
    train_set_images = np.concatenate((train_set[0], valid_set[0]), axis = 0)
    # 1d array, 60000 
    train_set_labels = np.concatenate((train_set[1], valid_set[1]))
    # transform digit in labels to ont-hot representation
    # becomes 2d array, 60000 x 10
    train_set_labels_vec = np.zeros((len(train_set_labels), 10))
    for i in range(len(train_set_labels)):
        train_set_labels_vec[i, train_set_labels[i]] = 1
    training_data = [] 
    training_data.append(train_set_images)
    training_data.append(train_set_labels_vec)
    
    test_set_images = test_set[0]
    test_set_labels = test_set[1]
    # transform digit in labels to ont-hot representation
    # becomes 2d array, 10000 x 10
    #test_set_labels_vec = np.zeros((len(test_set_labels), 10))
    #for i in range(len(test_set_labels)):
    #    test_set_labels_vec[i, test_set_labels[i]] = 1
    test_data = []
    test_data.append(test_set_images)
    test_data.append(test_set_labels)
    
    # You may also use the gui_template.py functions to collect image data from the user. eg:
    # training_data = gt.get_images()
    # if len(training_data) != 0:
        #     gt.visualize_image(training_data[-1])
    
    # If you wish, you can have a loop here, or any other place really, that will update the
    # parameters below automatically.
    
    # Initialize network(s) here
    # two components in the initialization parameters, network layer size and number of data
    # for the layer size, each number stands for the number of neurons in each layer
    # from left to right are input layer, hidden layer(s), output layer
    initialization_params = [[784, 10], training_data[0].shape[0]]       
    feedforward_classifier_state = None
    feedforward_classifier_connections = None 
    [feedforward_classifier_state, feedforward_classifier_connections] = init_feedforward_classifier(initialization_params)
    # Change initialization params if desired
    #initialization_params = [[784, 50, 784], training_data[0].shape[0]]       
    #autoencoder_state = None
    #autoencoder_connections = None
    #[autoencoder_state, autoencoder_connections] = init_autoencoder(initialization_params)
    # Change initialization params if desired
    #initialization_params = [[784, 50, 20, 10], training_data[0].shape[0], autoencoder_connections[0]]       
    #autoencoder_classifier_state = None
    #autoencoder_classifier_connections = None
    #[autoencoder_classifier_state, autoencoder_classifier_connections] = init_autoencoder_classifier(initialization_params)
    
    
    # Train network(s) here
    # learning rate, training iterations
    training_params = [0.00002, 400]
    feedforward_classifier_connections = train_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, training_data, test_data, training_params)
    feedforward_classifier_connections_json = [c.tolist() for c in feedforward_classifier_connections]
    # save feedforward classifier weights in json file
    print "saving feedforward classifier connections..."
    with open('fc_none_123_0001_400.json', 'w') as f:
        f.write(json.dumps(feedforward_classifier_connections_json))
    # Change training params if desired
    # learning rate, iteration number, sparse constraint, sparse rate
    #training_params = [0.000002, 100, 0.05, 1]
    #autoencoder_connections = train_autoencoder(autoencoder_state, autoencoder_connections, training_data, test_data, training_params)
    #autoencoder_connections_json = [c.tolist() for c in autoencoder_connections]
    ## save autoencoder weights in json file
    #print "saving autoencoder connections..."
    #with open('autoencoder_connections.json', 'w') as f:
    #    f.write(json.dumps(autoencoder_connections_json))
    ## Change training params if desired
    #training_params = [0.00002, 5]
    #autoencoder_classifier_connections = train_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, training_data, test_data, training_params)
    #autoencoder_classifier_connections_json = [c.tolist() for c in autoencoder_classifier_connections]
    ## save autoencoder classifier weights in json file
    #print "saving autoencoder classifier connections..."
    #with open('autoencoder_classifier_connections.json', 'w') as f:
    #    f.write(json.dumps(autoencoder_classifier_connections_json))
   
    
    ## Test network(s) here
    #test_params = None
    #accuracy = test_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, test_data)
    #print "Accuracy on test data: %f" % accuracy
    ## Change test params if desired
    #test_autoencoder(autoencoder_state, autoencoder_connections, test_data, test_params)
    ## Change test params if desired
    #test_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, test_data, test_params)
    #    
    ## You can use gui_template.py functions for visualization as you wish
