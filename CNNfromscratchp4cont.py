import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1, 2, 3, 2.5],
		[0.5, 5.0, -1.0, 2.0],
		[-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)
#100 feature sets of 3 inputs



class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #shapes (n inputs, n neurons)
		#why are we doing n inputs by n neurons?
		#in the last part, we did the number of neurons by the number of weights
		#when we go and do our forward pass, we won't have to do a transpose like we did before
		self.biases = np.zeros((1, n_neurons))
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases



class Activation_ReLU:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)
#print(layer1.output)

'''
layer1 = Layer_Dense(4,5)
#note that making this an object makes it a lot more dynamic than just manually making each neuron, with weights and biases. We can now easily construct neurons with the same structure
layer2 = Layer_Dense(5,2)
#input of layer2 MUST equal output of layer 1, again following the matrix multiplication property (A*B is only viable if A=B.T)
#meaning, layer 3 now must have an input of 2, and so forth

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)'''

#if we are initializing a new neural network, we need to assign weights and biases
#the way we initialize weights is as random values of -1 and +1, the smaller the range the better (we can go -0.1 to +0.1)
#the same can be said about biases, we tend to start with -0.1 to 0.1. If our outputs are 0, then the subsequent layers will also likely be 0, resulting in a "dead" network
#therefore, we would want to consider increasing the values
#generally, we want small values. The reason why is if you have the input > |1|, then each subsequent layer gets bigger and bigger
#resulting in a really large, computationally expensive neural network

#activation functions
#1) Heaviside/Step Function (0 or 1) literally outputs a 0 or 1 depending on the inputs of the weights and biases. The output of this becomes the input of the next neuron (which is either a 0 or 1)
#2) Sigmoid Function. Usually preferred because it allows for a more granular approach. This enables us to better estimate the error range of our neural network, and make the necessary adjustment more easily
#3) Rectified Linear Unit Function. If x>0, the output is x, if x <= 0, the output is 0. This is usually the best. Simple, granular, and fast.
#why do we use activation functions? If we were just to use weights a biases (y =x), we can only fit linear function. If we try to take a non-linear function and fit it into a linear function
#the approximation is a lot worse compared to ReLU
#why does it work? Everything is built into the fact that it's NOT LINEAR. ReLU, is so close to being linear,  but it's not. It actually is really easily to manipulate the neuron into whatever non-linear function we can approximate
#strongly recommend to watch the vide, 8:00-11:00 minute mark is when he talks about ReLU
#first neuron is responsible for setting the activation point, second is responsible for deactivation, third is another activation, and so on until we manipulate the function to mimick the non-linear function


#
