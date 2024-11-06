import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1, 2, 3, 2.5],
     [0.5, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)  # 100 feature sets of 3 inputs

# Layer_Dense class defines a layer with specified input and neuron count
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with a small random value, scaled for better initial convergence
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # shape (n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  # Initialize biases to 0
    def forward(self, inputs):
        # Calculate output values using dot product + bias
        self.output = np.dot(inputs, self.weights) + self.biases


# Activation_ReLU class applies the ReLU (Rectified Linear Unit) activation function
class Activation_ReLU:
    def forward(self, inputs):
        # ReLU sets all negative values to 0
        self.output = np.maximum(0, inputs)


# Activation_Softmax class applies the softmax activation function
class Activation_Softmax:
    def forward(self, inputs):
        # Subtract max value in each row for numerical stability (avoids large exponentials)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize exponentiated values to get probabilities that sum to 1 for each row
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities  # Softmax output represents the class probabilities


X, y = spiral_data(samples=100, classes=3)

# First dense layer with 2 inputs, 3 neurons
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# Second dense layer with 3 inputs (matching previous layer’s outputs), 3 neurons for 3-class output
dense2 = Layer_Dense(3, 3)
activaton2 = Activation_Softmax()

# Forward pass through first dense layer
dense1.forward(X)
# Activation function ReLU applied to first layer’s output
activation1.forward(dense1.output)

# Forward pass through second dense layer
dense2.forward(activation1.output)
# Softmax activation applied to second layer’s output
activaton2.forward(dense2.output)

# Print the output of softmax activation for first 5 samples
print(activaton2.output[:5])
