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

# Initialize feedforward classifier
def init_feedforward_classifier(initialization_params):
    # a list indicating size of each layer, including input and output layers
    layers_size = initialization_params[0]
    # total number of data, for MNIST, it has 60000 training data
    data_num = initialization_params[1]

    feedforward_classifier_state = [] 
    # initialize states of each layer with zeros
    # each layer's state is a matrix with size (data_num, layer_size), which stores states of all the training data
    for layer_size in layers_size:
        feedforward_classifier_state.append(np.zeros((data_num, layer_size)))

    feedforward_classifier_connections = []
    # initialize network weights with uniform random values
    for i in range(len(layers_size)-1):
        # plus one for the weights of bias
        num_row = layers_size[i] + 1
        num_col = layers_size[i+1]
        # random initialization with interval [-0.1, 0.1]
        feedforward_classifier_connections.append((np.random.rand(num_row, num_col) - 0.5) / 5)

    return [feedforward_classifier_state, feedforward_classifier_connections]

# Initialize autoencoder
def init_autoencoder(initialization_params):
    # a list indicating size of each layer, including input and output layers
    layers_size = initialization_params[0]
    # total number of data, for MNIST, it has 60000 training data
    data_num = initialization_params[1]

    autoencoder_state = [] 
    # initialize states of each layer with zeros
    # each layer's state is a matrix with size (data_num, layer_size), which stores states of all the training data
    for layer_size in layers_size:
        autoencoder_state.append(np.zeros((data_num, layer_size)))

    autoencoder_connections = []
    # initialize network weights with uniform random values
    for i in range(len(layers_size)-1):
        # plus one for the weights of bias
        num_row = layers_size[i] + 1
        num_col = layers_size[i+1]
        # random initialization with interval [-0.1, 0.1]
        autoencoder_connections.append((np.random.rand(num_row, num_col) - 0.5) / 5)

    return [autoencoder_state, autoencoder_connections]

# Initialize autoencoder classifier
def init_autoencoder_classifier(initialization_params):
    # a list indicating size of each layer, including input and output layers
    layers_size = initialization_params[0]
    # total number of data, for MNIST, it has 60000 training data
    data_num = initialization_params[1]
    # the weights of a trained hidden layer from autoencoder
    first_layer_connection = initialization_params[2]

    autoencoder_classifier_state = [] 
    # initialize states of each layer with zeros
    # each layer's state is a matrix with size (data_num, layer_size), which stores states of all the training data
    for layer_size in layers_size:
        autoencoder_classifier_state.append(np.zeros((data_num, layer_size)))

    autoencoder_classifier_connections = []
    # initialize the weights of the first layer with trained weights from autoencoder
    autoencoder_classifier_connections.append(first_layer_connection)
    # initialize network weights with uniform random values except the first layer
    for i in range(1, len(layers_size)-1):
        # plus one for the weights of bias
        num_row = layers_size[i] + 1
        num_col = layers_size[i+1]
        # random initialization with interval [-0.1, 0.1]
        autoencoder_classifier_connections.append((np.random.rand(num_row, num_col) - 0.5) / 5)

    return [autoencoder_classifier_state, autoencoder_classifier_connections]
    
# Sigmoid activation function, the input can be an array type matrix
def sigmoid(o):
    return 1.0 / (1 + np.exp(-o))

# Update function of feedforward classifier, given input, update all states forwardly
def update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections):
    # total number of data, for MNIST, it has 60000 training data
    data_num = feedforward_classifier_state[0].shape[0]
    # bias is always one 
    bias = np.ones((data_num, 1))
    # iteratively update each layer
    for i in range(len(feedforward_classifier_state)-1):
        # combine bias with other neurons of this layer
        state_temp = np.concatenate((feedforward_classifier_state[i], bias), axis = 1)
        # matrix multiplication with corresponding weights
        feedforward_classifier_state[i+1] = state_temp.dot(feedforward_classifier_connections[i])
        # each state then transformed by sigmoid function
        feedforward_classifier_state[i+1] = sigmoid(feedforward_classifier_state[i+1])

    return feedforward_classifier_state
    
# Update function of autoencoder, given input, update all states forwardly
def update_autoencoder(autoencoder_state, autoencoder_connections):
    # total number of data, for MNIST, it has 60000 training data
    data_num = autoencoder_state[0].shape[0]
    # bias is always one 
    bias = np.ones((data_num, 1))
    # iteratively update each layer
    for i in range(len(autoencoder_state)-1):
        # combine bias with other neurons of this layer
        state_temp = np.concatenate((autoencoder_state[i], bias), axis = 1)
        # matrix multiplication with corresponding weights
        autoencoder_state[i+1] = state_temp.dot(autoencoder_connections[i])
        # each state then transformed by sigmoid function
        autoencoder_state[i+1] = sigmoid(autoencoder_state[i+1])

    return autoencoder_state
    
# Update function of autoencoder classifier, given input, update all states forwardly
def update_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections):
    # total number of data, for MNIST, it has 60000 training data
    data_num = autoencoder_classifier_state[0].shape[0]
    # bias is always one 
    bias = np.ones((data_num, 1))
    # iteratively update each layer
    for i in range(len(autoencoder_classifier_state)-1):
        # combine bias with other neurons of this layer
        state_temp = np.concatenate((autoencoder_classifier_state[i], bias), axis = 1)
        # matrix multiplication with corresponding weights
        autoencoder_classifier_state[i+1] = state_temp.dot(autoencoder_classifier_connections[i])
        # each state then transformed by sigmoid function
        autoencoder_classifier_state[i+1] = sigmoid(autoencoder_classifier_state[i+1])

    return autoencoder_classifier_state
    
# Backpropagation for feedforward classifier
def bp_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, target, training_params):
    # total number of data, for MNIST, it has 60000 training data
    data_num = feedforward_classifier_state[0].shape[0]
    connections_num = len(feedforward_classifier_connections)
    bias = np.ones((data_num, 1))
    # a list for storing gradients of each weight matrix
    gradients = []
    # a list for storing accumulated errors of each layer
    deltas = []
    # iteratively compute cumulated errors and gradient of each layer
    for i in range(connections_num):
        output = feedforward_classifier_state[connections_num-i]
        # compute accumulated error of output layer
        if i == 0:
            error = output - target
            delta = output * (1 - output) * error
        # compute accumulated error of hidden layer
        else:
            delta = output * (1 - output) * deltas[-1].dot(feedforward_classifier_connections[connections_num-i][:-1,:].T)
        deltas.append(delta)
        # get the output from the previous layer
        output_pre = feedforward_classifier_state[connections_num-i-1]
        output_pre = np.concatenate((output_pre, bias), axis = 1)
        # compute gradient for this layer
        gradient = - training_params[0] * output_pre.T.dot(delta)
        gradients.append(gradient)

    # the gradients are computed backward, so reverse it
    return gradients[::-1]

# Backpropagation for autoencoder
def bp_autoencoder(autoencoder_classifier_state, autoencoder_classifier_connections, target, training_params):
    # total number of data, for MNIST, it has 60000 training data
    data_num = autoencoder_classifier_state[0].shape[0]
    connections_num = len(autoencoder_classifier_connections)
    bias = np.ones((data_num, 1))
    # autoencoder can be trained with or without sparse constraint
    use_sparse = training_params[4]
    if use_sparse == True:
        # desired average activation value on hidden layer, usually close to zero
        rho = training_params[2]
        # penalty term of the sparse constraint, similar to learning rate
        beta = training_params[3]
        # real average activation value on hidden layer
        rho_hat = np.mean(autoencoder_state[1], axis=0)
    # a list for storing gradients of each weight matrix
    gradients = []
    # a list for storing accumulated errors of each layer
    deltas = []
    # iteratively compute cumulated errors and gradient of each layer
    for i in range(connections_num):
        output = autoencoder_state[connections_num-i]
        # compute accumulated error of output layer
        if i == 0:
            error = output - target
            delta = output * (1 - output) * error
        # compute accumulated error of hidden layer
        else:
            if use_sparse == True:
                sparse_penalty = beta * (-1.0 * rho / rho_hat + (1.0 - rho) / (1.0 - rho_hat))
                delta =  sparse_penalty + deltas[-1].dot(autoencoder_connections[connections_num-i][:-1,:].T)
                delta = delta * output * (1.0 - output)
            else:
                delta = output * (1 - output) * deltas[-1].dot(autoencoder_connections[connections_num-i][:-1,:].T)
        deltas.append(delta)
        # get the output from the previous layer
        output_pre = autoencoder_state[connections_num-i-1]
        output_pre = np.concatenate((output_pre, bias), axis = 1)
        # compute gradient for this layer
        gradient = - training_params[0] * output_pre.T.dot(delta)
        gradients.append(gradient)

    # the gradients are computed backward, so reverse it
    return gradients[::-1]

# Backpropagation for autoencoder classifier
def bp_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, target, training_params):
    # total number of data, for MNIST, it has 60000 training data
    data_num = autoencoder_classifier_state[0].shape[0]
    connections_num = len(autoencoder_classifier_connections)
    bias = np.ones((data_num, 1))
    # whether update the first layer weights or not
    fine_tune = training_params[2]
    # a list for storing gradients of each weight matrix
    gradients = []
    # a list for storing accumulated errors of each layer
    deltas = []
    # also update the first layer weights
    if fine_tune == True:
        for i in range(connections_num):
            output = autoencoder_classifier_state[connections_num-i]
            # compute accumulated error of output layer
            if i == 0:
                error = output - target
                delta = output * (1.0 - output) * error
            # compute accumulated error of hidden layer
            else:
                delta = output * (1.0 - output) * deltas[-1].dot(autoencoder_classifier_connections[connections_num-i][:-1,:].T)
            deltas.append(delta)
            # get the output from the previous layer
            output_pre = autoencoder_classifier_state[connections_num-i-1]
            output_pre = np.concatenate((output_pre, bias), axis = 1)
            # compute gradient for this layer
            gradient = - training_params[0] * output_pre.T.dot(delta)
            gradients.append(gradient)
    # do not update the first layer weights
    else:
        for i in range(connections_num - 1):
            output = autoencoder_classifier_state[connections_num-i]
            # compute accumulated error of output layer
            if i == 0:
                error = output - target
                delta = output * (1.0 - output) * error
            # compute accumulated error of hidden layer
            else:
                delta = output * (1.0 - output) * deltas[-1].dot(autoencoder_classifier_connections[connections_num-i][:-1,:].T)
            deltas.append(delta)
            # get the output from the previous layer
            output_pre = autoencoder_classifier_state[connections_num-i-1]
            output_pre = np.concatenate((output_pre, bias), axis = 1)
            # compute gradient for this layer
            gradient = - training_params[0] * output_pre.T.dot(delta)
            gradients.append(gradient)

    # the gradients are computed backward, so reverse it
    return gradients[::-1]

# Cost function of feedforward classifier
def cost_feedforward_classifier(output, target):
    cost = np.sum(np.square(output - target)) / 2

    return cost

# Cost function of autoencoder
def cost_autoencoder(autoencoder_state, target, training_params):
    output = autoencoder_state[-1]
    use_sparse = training_params[4]
    # if use sparse, also add the sparse constraint as cost
    if use_sparse == True:
        rho = training_params[2]
        rho_hat = np.mean(autoencoder_state[1], axis=0)
        cost1 = np.sum(np.square(output - target)) / 2
        cost2 = rho * np.log(1.0 * rho / rho_hat) + (1.0 - rho) * np.log((1.0-rho) / (1.0 - rho_hat))
        cost2 = np.sum(cost2)

        return cost1 + cost2
    else:
        cost = np.sum(np.square(output - target)) / 2

        return cost
        
# Cost function of autoencoder classifier
def cost_autoencoder_classifier(output, target):
    cost = np.sum(np.square(output - target)) / 2

    return cost

# Main training function of feedforward classifier
def train_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, training_data, test_data, training_params):
    print "Start training feedforward classifier..."
    data_num = training_data[0].shape[0]
    # number of iterations for training, this is the only stop criterion
    iteration_num = training_params[1]
    costs = []
    for i in range(iteration_num):
        feedforward_classifier_state[0] = training_data[0]
        feedforward_classifier_state = update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections)
        cost = cost_feedforward_classifier(feedforward_classifier_state[-1], training_data[1])
        costs.append(cost)
        print "iteration %d, cost: %f" % (i, cost)
        # compute gradients using backpropagation
        gradients = bp_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, training_data[1], training_params)
        # update weights using the gradients
        for k in range(len(gradients)):
            feedforward_classifier_connections[k] += gradients[k]
    # print the final cost on training data
    feedforward_classifier_state[0] = training_data[0]
    feedforward_classifier_state = update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections)
    cost = cost_feedforward_classifier(feedforward_classifier_state[-1], training_data[1])
    costs.append(cost)
    print "iteration %d, cost: %f" % (iteration_num, cost)
    print "Finish training!"

    return feedforward_classifier_connections
    
# Main training function of autoencoder
def train_autoencoder(autoencoder_state, autoencoder_connections, training_data, test_data, training_params):
    use_sparse = training_params[4]
    if use_sparse == True:
        print "Start training sparse autoencoder..."
    else:
        print "Start training autoencoder..."
    data_num = training_data[0].shape[0]
    # number of iterations for training, this is the only stop criterion
    iteration_num = training_params[1]
    costs = []
    for i in range(iteration_num):
        autoencoder_state[0] = training_data[0]
        autoencoder_state = update_autoencoder(autoencoder_state, autoencoder_connections)
        cost = cost_autoencoder(autoencoder_state, training_data[0], training_params)
        costs.append(cost)
        print "iteration %d, cost: %f" % (i, cost)
        # compute gradients using backpropagation
        gradients = bp_autoencoder(autoencoder_state, autoencoder_connections, training_data[0], training_params)
        # update weights using the gradients
        for k in range(len(gradients)):
            autoencoder_connections[k] += gradients[k]
    # print the final cost on training data
    autoencoder_state[0] = training_data[0]
    autoencoder_state = update_autoencoder(autoencoder_state, autoencoder_connections)
    cost = cost_autoencoder(autoencoder_state, training_data[0], training_params)
    costs.append(cost)
    print "iteration %d, cost: %f" % (iteration_num, cost)
    print "Finish training!"
    
    return autoencoder_connections

# Main training function of autoencoder classifier
def train_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, training_data, test_data, training_params):
    print "Start training autoencoder classifier..."
    data_num = training_data[0].shape[0]
    # number of iterations for training, this is the only stop criterion
    iteration_num = training_params[1]
    fine_tune = training_params[2]
    costs = []
    for i in range(iteration_num):
        autoencoder_classifier_state[0] = training_data[0]
        autoencoder_classifier_state = update_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections)
        cost = cost_autoencoder_classifier(autoencoder_classifier_state[-1], training_data[1])
        costs.append(cost)
        print "iteration %d, cost: %f" % (i, cost)
        # compute gradients using backpropagation
        gradients = bp_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, training_data[1], training_params)
        # update weights using the gradients
        if fine_tune == True:
            for k in range(len(gradients)):
                autoencoder_classifier_connections[k] += gradients[k]
        else:
            for k in range(len(gradients)):
                autoencoder_classifier_connections[k+1] += gradients[k]
    # print the final cost on training data
    autoencoder_classifier_state[0] = training_data[0]
    autoencoder_classifier_state = update_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections)
    cost = cost_autoencoder_classifier(autoencoder_classifier_state[-1], training_data[1])
    costs.append(cost)
    print "iteration %d, cost: %f" % (iteration_num, cost)
    print "Finish training!"
    
    return autoencoder_classifier_connections

# Test function of feedforward classifier, outputs accuracy on test data
def test_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, test_data):
    # inputs the test data and update the network states
    feedforward_classifier_state[0] = test_data[0]
    feedforward_classifier_state = update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections)
    # do prediction as the class with the largest output
    output = np.argmax(feedforward_classifier_state[-1], axis = 1)
    # labels of the test data
    target = test_data[1]
    # count correct predictions
    correct = list(output - target).count(0)
    
    return correct / len(test_data[1])

# Test function of feedforward classifier, outputs cost on test data
def test_autoencoder(autoencoder_state, autoencoder_connections, test_data, training_params):
    # inputs the test data and update the network states
    autoencoder_state[0] = test_data[0]
    autoencoder_state = update_autoencoder(autoencoder_state, autoencoder_connections)
    output = autoencoder_state[-1]
    target = test_data[0]
    # for autoencoder, compute cost on test data instead of do predictions
    cost_test = cost_autoencoder(autoencoder_state, target, training_params)
    
    return cost_test

# Test function of feedforward classifier, outputs accuracy on test data
def test_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, test_data):
    # inputs the test data and update the network states
    autoencoder_classifier_state[0] = test_data[0]
    autoencoder_classifier_state = update_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections)
    # do prediction as the class with the largest output
    output = np.argmax(autoencoder_classifier_state[-1], axis = 1)
    # labels of the test data
    target = test_data[1]
    # count correct predictions
    correct = list(output - target).count(0)
    
    return correct / len(test_data[1])


if __name__=='__main__':

    # Please use the following snippet for clarity if you have command-line 
    # arguments:

    # if len(argv) < <number_of_expected_arguments>:
    #    print('Usage: python', argv[0], '<expected_arg_1>', '<expected_arg_2>, ...')
    #    exit(1)
    
    # arg_1 = argv[1]
    # arg_2 = argv[2]
    # ...
    
    # Read data from a preprocessed MNIST data
    print "Reading MNIST data...\n"
    f = gzip.open('mnist.pkl.gz', 'rb')
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
    # form the complete training data
    training_data = [] 
    training_data.append(train_set_images)
    training_data.append(train_set_labels_vec)
    
    # 2d array, 10000 x 784
    test_set_images = test_set[0]
    # 1d array, 10000 
    test_set_labels = test_set[1]
    # form the complete test data
    test_data = []
    test_data.append(test_set_images)
    test_data.append(test_set_labels)
    
    ###############################################################################################################
    # For feedforward classifier, just comment this block if you don't use feedforward network
    # Initialization
    # two components in the initialization parameters, network layer size and number of data
    # for the layer size, each number stands for the number of neurons in each layer
    # from left to right are input layer, hidden layer(s), output layer
    # e.g. no hidden layer: [784, 10], two hidden layers: [784, 100, 50, 10]
    initialization_params = [[784, 100, 50, 10], training_data[0].shape[0]]       
    feedforward_classifier_state = None
    feedforward_classifier_connections = None 
    [feedforward_classifier_state, feedforward_classifier_connections] = init_feedforward_classifier(initialization_params)
    # Training 
    # parameters for training: learning rate, training iterations
    training_params = [0.00003, 800]
    feedforward_classifier_connections = train_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, training_data, test_data, training_params)
    # save feedforward classifier weights in json file
    print "Saving feedforward classifier connections..."
    feedforward_classifier_connections_json = [c.tolist() for c in feedforward_classifier_connections]
    # generate file name
    nn = 'fc'
    if len(initialization_params[0]) > 2:
        hidden_layer = '-'.join([str(x) for x in initialization_params[0][1:-1]])
    else:
        hidden_layer = 'none'
    seed = '123'
    learning_rate = format(training_params[0], 'f')
    iteration_num = str(training_params[1])
    file_name = '_'.join([nn, hidden_layer, seed, learning_rate, iteration_num]) + '.json'
    with open(file_name, 'w') as f:
        f.write(json.dumps(feedforward_classifier_connections_json))
    # Testing
    accuracy = test_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, test_data)
    print "Feedforward classifier, accuracy on test data: %f\n" % accuracy
    ###############################################################################################################

    ###############################################################################################################
    # For autoencoder, just comment this block if you don't use autoencoder
    # Initialization
    initialization_params = [[784, 100, 784], training_data[0].shape[0]]       
    autoencoder_state = None
    autoencoder_connections = None
    [autoencoder_state, autoencoder_connections] = init_autoencoder(initialization_params)
    # Training 
    # parameters for training:
    # learning rate, iteration number, sparse constraint on activated hidden neuron, sparse penalty weight, use sparse or not
    training_params = [0.00003, 600, 0.05, 1, False]
    autoencoder_connections = train_autoencoder(autoencoder_state, autoencoder_connections, training_data, test_data, training_params)
    # save autoencoder weights in json file
    print "Saving autoencoder connections..."
    autoencoder_connections_json = [c.tolist() for c in autoencoder_connections]
    # generate file name
    nn = 'ae'
    hidden_layer = '-'.join([str(x) for x in initialization_params[0][1:-1]])
    seed = '123'
    learning_rate = format(training_params[0], 'f')
    iteration_num = str(training_params[1])
    if training_params[4] == True:
        sparse = 'sparse'
        file_name = '_'.join([nn, hidden_layer, seed, learning_rate, iteration_num, sparse]) + '.json'
    else:
        file_name = '_'.join([nn, hidden_layer, seed, learning_rate, iteration_num]) + '.json'
    with open(file_name, 'w') as f:
        f.write(json.dumps(autoencoder_connections_json))
    # Testing
    cost_test = test_autoencoder(autoencoder_state, autoencoder_connections, test_data, training_params)
    print "Autoencoder, cost on test data: %f\n" % cost_test
    ###############################################################################################################

    ###############################################################################################################
    # For autoencoder classifier, just comment this block if you don't use autoencoder classifier
    # Initialization
    # load trained weights of autoencoder
    print "Loading weights from trained autoencoder..."
    with open('ae_100-50_123_0.000030_600.json', 'r') as f:
        autoencoder_connections = json.loads(f.read())
    autoencoder_connections = [np.array(c) for c in autoencoder_connections]
    # initialization parameters: network structure, number of training data, trained weights of autoencoder
    initialization_params = [[784, 100, 50, 10], training_data[0].shape[0], autoencoder_connections[0]]       
    autoencoder_classifier_state = None
    autoencoder_classifier_connections = None
    [autoencoder_classifier_state, autoencoder_classifier_connections] = init_autoencoder_classifier(initialization_params)
    # Training 
    # learning rate, iteration number, fine-tune the first layer or not
    training_params = [0.00003, 800, False]
    autoencoder_classifier_connections = train_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, training_data, test_data, training_params)
    # save autoencoder classifier weights in json file
    print "Saving autoencoder classifier connections..."
    autoencoder_classifier_connections_json = [c.tolist() for c in autoencoder_classifier_connections]
    # generate file name
    nn = 'ac'
    hidden_layer = '-'.join([str(x) for x in initialization_params[0][1:-1]])
    seed = '123'
    learning_rate = format(training_params[0], 'f')
    iteration_num = str(training_params[1])
    if training_params[2] == True:
        fine_tune = 'fine-tune'
        file_name = '_'.join([nn, hidden_layer, seed, learning_rate, iteration_num, fine_tune]) + '.json'
    else:
        file_name = '_'.join([nn, hidden_layer, seed, learning_rate, iteration_num]) + '.json'
    with open(file_name, 'w') as f:
        f.write(json.dumps(autoencoder_classifier_connections_json))
    # Testing
    accuracy = test_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, test_data)
    print "Autoencoder classifier, accuracy on test data: %f" % accuracy
    ###############################################################################################################
