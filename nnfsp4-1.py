import numpy as np

#4 inputs, 3 neurons, meaning 3 unique weight sets, and 3 unique biases
inputs = [[1, 2, 3, 2.5],
		[0.5, 5.0, -1.0, 2.0],
		[-1.5, 2.7, 3.3, -0.8]]

#weights for neurons 1, 2, 3 respectively (provides list of strength of connections)
#each item in the array = 1 input
weights= [[0.2, 0.8, -0.5, 1.0],
		  [0.5, -0.91, 0.26, -0.5],
		  [-0.26, -0.27, 0.17, 0.87]]

#biases for neurons 1,2,3 respectively (used as an 'offset' to the weights)
biases = [2, 3, 0.5]

#weights for neurons 1, 2, 3 respectively (provides list of strength of connections)
#each item in the array = 1 input
weights2 = [[0.1, -0.14, 0.5],
		  [-0.5, 0.12, -0.33],
		  [-0.44, 0.73, -0.13]]

#biases for neurons 1,2,3 respectively (used as an 'offset' to the weights)
biases2 = [-1, 2, -0.5]
#how do we change the output?

#well, we can't really change the inputs, that's sort of given, so we must use the weights and biases to do so.
#so the main job is to figure out how to tune the weights and our biases for optimal results
#we can think of weights and biases as 'knobs' that the optimizer tunes 

#why are they different?
#say we have some value -0.5, weight and bias = 0.7
#if we print (some_value*weight), we get -0.35

#if we print (some_value+bias), we get ~0.2
#bias OFFSETS the value
#weights change the magnitude
#if we think of this as y = mx + b, m is the weights, b is the bias here

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
#note if we were to do the dot product of inputs & weights, we would get a shape error
#because we want things indexed by the neurons (weight sets)

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)

#shape: for each array, the size of the array needs to be the same. An array contained within an array is a shape
#for instance, if we have a list of lists (lol), given by [[1, 5, 6, 2], [3, 2, 1, 3]], our shape is given by (2, 4)
#if we have a lolol, it would be given by ([# of elements in parent list], [# of elements in child list], [# of elements in grandchild list])
#it follows the same procedure for any nth dimensioned shape
#tensor: an object that CAN (doesn't have to be) be represented as an array
#in deep learning, a tensor is usually an array


#BATCHES
#the reason why we use batches is we can calculate things in parallel
#note: we use GPUs for training models because CPUs have 4-8 cores, GPUs have thousands
#CPUs are mostly used for complex calculations, since we're doing relatively basic computations like matrix multiplication, a GPU is ideal
#additionally, batches help with generalization. Instead of doing one example at a time, we can do multiple at a time
#to visualize this, think of a line of best fit
#we don't want to make a line of best fit for each batch, if we can do multiple batches we can get a better generalization and less computation
#it becomes much easier to draw lines of best fit for 4, 16, 32 batches at a time vs 1 batch at a time
#but sometimes there's too much, because there is overfitting, we need to find an equilibrium
#32 is usually a common size

#with batches, we're doing matrix multiplication, (dot product but just higher level)

#if we were to do the matrix multiplication like before, we would get a shape error
#why? because the inputs array has a shape of (3,4), because it has 3 lists, and each list has 4 elements
#the weights array also has a shape of (3,4)
#a property of matrix multiplication is for matrices A,B, we can only do matrix multiplication if dim(A) = dim(B.T) (.T is transpose)
#because for us to generate the mxn matrix, we need to have the number of columns equal the number of rows
#the reason why we do inputs by weights, it because inputs represent data points or features (nodes)
#the weights are the connections to the nodes
#that way, the outputs give us each neuron's response to the input, which is the desired output

#for another layer, we need another set of weights and biases