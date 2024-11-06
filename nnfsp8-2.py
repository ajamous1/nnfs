import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1, 2, 3, 2.5],
     [0.5, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# Generate a dataset of 100 samples with 3 classes
X, y = spiral_data(100, 3)  # 100 feature sets of 3 inputs

# Layer_Dense class represents a fully connected layer in the network
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # shape (n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  # shape (1, n_neurons)
    def forward(self, inputs):
        # Compute the layer's output using dot product and adding bias
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation function class
class Activation_ReLU:
    def forward(self, inputs):
        # Apply ReLU function, setting all negative values to 0
        self.output = np.maximum(0, inputs)

# Softmax activation function class
class Activation_Softmax:
    def forward(self, inputs):
        # Exponentiate values after subtracting max value for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize to get probabilities, summing to 1 across each row
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Base loss class to calculate mean loss over all samples
class Loss:
    def calculate(self, output, y):
        # Calculate individual losses for each sample
        sample_losses = self.forward(output, y)
        # Average loss across samples
        data_loss = np.mean(sample_losses)
        return data_loss

# Categorical Cross-Entropy Loss class for multi-class classification
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # Number of samples in batch
        samples = len(y_pred)
        # Clip predictions to avoid log(0) error (small values get clipped)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Handle different formats of target values (sparse or one-hot encoded)
        # If labels are sparse (1D), select the confidence of the correct class
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Ig labels are one-hot encoded (2D), take dot product with predictions to get correct confidences
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Calculate negative log likelihood for each sample
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

# Initialize dataset with 100 samples and 3 classes
X, y = spiral_data(samples=100, classes=3)

# Create the layers and activations
dense1 = Layer_Dense(2, 3)  # First dense layer, 2 inputs (features), 3 neurons
activation1 = Activation_ReLU()  # ReLU activation for first layer

dense2 = Layer_Dense(3, 3)  # Second dense layer, 3 inputs (neurons), 3 neurons (one for each class)
activation2 = Activation_Softmax()  # Softmax activation for output layer to produce probabilities

# Forward pass through first layer and ReLU activation
dense1.forward(X)
activation1.forward(dense1.output)

# Forward pass through second layer and Softmax activation
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Print first 5 samples of the softmax output (class probabilities)
print(activation2.output[:5])

# Calculate and print the loss
loss_function = Loss_CategoricalCrossentropy()  # Initialize categorical cross-entropy loss function
loss = loss_function.calculate(activation2.output, y)  # Calculate mean loss for the batch

print(loss)
