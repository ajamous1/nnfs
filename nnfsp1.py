inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3.0

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)

#This is effectively an neuron with 3 inputs. It takes in all the connections from the previous inputs based on their respective weights to come up with an output value summed with the bias