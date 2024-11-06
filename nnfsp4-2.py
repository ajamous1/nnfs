import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        #why are we doing n inputs by n neurons?
        #in the last part, we did the number of neurons by the number of weights
        #when we go and do our forward pass, we won't have to do a transpose like we did before

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)


#note that making this an object makes it a lot more dynamic than just manually making each neuron, with weights and biases. We can now easily construct neurons with the same structure

#input of layer2 MUST equal output of layer 1, again following the matrix multiplication property (A*B is only viable if A=B.T)
#meaning, layer 3 now must have an input of 2, and so forth
#if we find that all the outputs are approaching 0, or "the network is dead" one of the first things we should consider is to initalize the biases to a non-zero number


#if we are initializing a new neural network, we need to assign weights and biases
#the way we initialize weights is as random values of -1 and +1, the smaller the range the better (we can go -0.1 to +0.1)
#the same can be said about biases, we tend to start with -0.1 to 0.1. If our outputs are 0, then the subsequent layers will also likely be 0, resulting in a "dead" network
#therefore, we would want to consider increasing the values
#generally, we want small values. The reason why is if you have the input > |1|, then each subsequent layer gets bigger and bigger
#resulting in a really large, computationally expensive neural network