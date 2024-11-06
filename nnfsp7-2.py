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

class Activation_Softmax:
	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities

class Loss:
	def calculate(self, output, y):
		sample_losses = self.forward(output, y)
		data_loss = np.mean(sample_losses)
		return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-6, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods


X,y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activaton2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activaton2.forward(dense2.output)

print(activaton2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activaton2.output, y)

print(loss)
