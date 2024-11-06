import numpy as np 


inputs = [1.0, 2.0, 3.0, 2.5]

#weights for neurons 1, 2, 3 respectively (provides list of strength of connections)
#each item in the array = 1 input
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

#biases for neurons 1,2,3 respectively (used as an 'offset' to the weights)
biases = [2.0, 3.0, 0.5]

output = np.dot(weights, inputs) + biases
print(output)

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

#note if we were to do the dot product of inputs & weights, we would get a shape error
#because we want things indexed by the neurons (weight sets)

#shape: for each array, the size of the array needs to be the same. An array contained within an array is a shape
#for instance, if we have a list of lists (lol), given by [[1, 5, 6, 2], [3, 2, 1, 3]], our shape is given by (2, 4)
#if we have a lolol, it would be given by ([# of elements in parent list], [# of elements in child list], [# of elements in grandchild list])
#it follows the same procedure for any nth dimensioned shape
#tensor: an object that CAN (doesn't have to be) be represented as an array
#in deep learning, a tensor is usually an array