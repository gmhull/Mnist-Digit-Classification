import numpy as np
import os

class Perceptron(object):
    """docstring for Perceptron.  Creates a single perceptron with multiple inputs and a bias.

    Attributes:
        inputs: The number of inputs given to the perceptron. Does not include the bias.
        bias: The bias for each perceptron. Defaults to 1.0. """

    def __init__(self, inputs, bias=1.0):
        """Create a perceptron with a given number of inputs and a bias."""
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias
        #  Are we really adding a bias to the weights?

    def activate(self, x):
        """Take the inputs and bias to produce the output of the Perceptron."""
        sum = np.dot(np.append(x,self.bias),self.weights)
        return self.sigmoid(sum)

    def create_weights(self, init_weights):
        """"Use this function to assign known weights to the perceptron."""
        self.weights = np.array(init_weights)

    def sigmoid(self, x):
        """Evaluate the perceptron function for an input, x."""
        return 1 / (1 + np.exp(-x))

class Multilayer_Perceptron(object):
    """docstring for Multilayer_Perceptron.  Creates a single perceptron with multiple inputs and a bias.

    Attributes:
        layers: A python list detailing the number of elements in each layer.
        bias: The bias term. The same bias used for all
        eta: The learning rate of the system. """

    def __init__(self, layers, bias=1.0, eta=0.5):
        self.layers = np.array(layers, dtype=object) # Length is the number of layers, number is the perceptrons per layer
        self.bias = bias
        self.eta = eta
        self.network = []  # The list of neurons
        self.values = []   # The list of outputs
        self.d = []        # The list of the error terms (lowercase delta)

        for i in range(len(self.layers)):
            self.values.append([])  # Add a blank location for each layer
            self.d.append([])       # Add a blank location for each layer
            self.network.append([]) # Add a blank location for each layer
            self.values[i] = [0.0 for j in range(self.layers[i])] # Create 0 values for each perceptron
            self.d[i] = [0.0 for j in range(self.layers[i])]      # Create 0 values for each perceptron
            if i > 0: # the first layer is the input layer so it does not have any
                for j in range(self.layers[i]):
                    # Create an object of the perceptron class for every position in the network
                    self.network[i].append(Perceptron(inputs = self.layers[i-1], bias = self.bias))

        # Make an array of the data.
        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.values], dtype=object)
        self.d = np.array([np.array(x) for x in self.d], dtype=object)

    def setWeights(self, init_weights):
        """Set the weights of all perceptrons.
        init_weights is a list of lists that holds the weights of all but the input layer."""
        for i in range(len(init_weights)):
            for j in range(len(init_weights[i])):
                # The i+1 is used to not affect the initial input layer.
                self.network[i+1][j].create_weights(init_weights[i][j])

    def printWeights(self):
        """Print the weights given to each perceptron."""
        print()
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                # Print out the weights of each perceptron
                print("Layer: %d Neuron: %d - " % (i+1, j), self.network[i][j].weights)
        print()

    def saveWeights(self, file):
        with open(file, 'w') as save_weight_file:
            for i in range(1,len(self.network)):
                for j in range(self.layers[i]):
                    for k in self.network[i][j].weights:
                        save_weight_file.writelines('%s\n' % k)
                    # save_weight_file.write('\n')

    def readWeights(self, file):
        weights_array = []
        done = 0
        if os.stat(file).st_size == 0:
            raise ValueError("No Weights Detected")
        with open(file, 'r') as read_weight_file:
            data = read_weight_file.readlines()

            for i in range(1,len(self.network)): # 1 to 2
                weights_array.append([]) # Creates an array for each
                for j in range(self.layers[i]): # 1 to 10
                    weights_array[i-1].append([])
                    k = data[:self.layers[i-1]+1]
                    for line in k:
                        weights_array[i-1][j].append(float(line[:-2]))
                    data = data[len(k):]
        return weights_array


    def run(self, x):
        """Feed a sample x into the MultiLayer Perceptron.
        x is a list of the inputs to the network."""
        # Make an array of the data
        x = np.array(x, dtype=object)
        # Set the first layer of values to be the inputs, x.
        self.values[0] = x
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                # Assign the value to be equal to the output of the perceptron activation function
                self.values[i][j] = self.network[i][j].activate(self.values[i-1])
        # Return the output values of the network
        return self.values[-1]

    def backpropagation(self, x, y):
        """Run an (x, y) pair through the backpropagation algorithm"""
        x = np.array(x, dtype=object)
        y = np.array(y, dtype=object)

        # Step by step backpropagation
        MSE = self.MeanSquaredError(x, y)

        # Step 4: Calculate the error terms of each term
        for i in reversed(range(1,len(self.network)-1)):
            for h in range(len(self.network[i])):
                fwd_error = 0.0
                for k in range(self.layers[i+1]):
                    fwd_error += self.network[i+1][k].weights[h] * self.d[i+1][k]
                self.d[i][h] = self.values[i][h] * (1-self.values[i][h]) * fwd_error

        # Step 5 & 6: Calculate the deltas and update the weights
        for i in range(1,len(self.network)): # For all layer except for the input layer
            for j in range(self.layers[i]): # The number of perceptrons in each layer
                for k in range(self.layers[i-1]+1): # Iterate through the weights plus the bias term
                    if k == self.layers[i-1]: # This is to update the bias
                        delta = self.eta * self.d[i][j] * self.bias
                    else: # This is to update the weights
                        delta = self.eta * self.d[i][j] * self.values[i-1][k]
                    self.network[i][j].weights[k] += delta
        return MSE

    def MeanSquaredError(self, x, y):
        # Step 1: Feed a sample into the network
        outputs = self.run(x)

        # Step 2: Calculate the mean squared error (MSE)
        error = (y - outputs)
        MSE = sum( error ** 2 ) / self.layers[-1]

        # Step 3: Calculate the output error terms
        self.d[-1] = outputs * (1 - outputs) * error

        return MSE
