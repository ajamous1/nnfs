import math
import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
				 [8.9, -1.81, 0.2],
				 [1.41, 1.051, 0.026]]

E = math.e

exp_values = np.exp(layer_outputs)



print(np.sum(layer_outputs, axis=1, keepdims=True)) #axis=1 gives you the sum of each layer in the batch

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)
print(sum(norm_values))
#softmax activation

#the reason why we might want to consider softmax, is if we were to come to a conclusion on our neural network and we want to come to an output based on the probability distribution. If
#the inputs are negative, then ReLU is automatically gonna clip to 0, and therefore the distribution is going to be 100/0, which is not desirable
#if they're both negative, they're both going to be 0, so we have a dead neural network
#so to be able to solve this problem of negativity we use the exponential function
#we want to lose the negative, but not the meaning behind the negative, which is why the exponential function is ideal here
#e^-10 < e, e^10 > e, so we now have a way to maintain the meaning of the negative without actually having a negative number

#once we've exponentiated them, we want to normalize the values, to get the probability values, so that we can get a value of 1

#to summarize, input -> exponentiate -> normalize -> output. The exponentiation and normalization is the softmax function

#one slight issue with exponentiation functions is that the values explode grows. It doesn't take much to overflow. 
#To combat this we can do v = u - maxu, so we subtract the largest value from all of the values in that layer, causing the largest value to be a 0
#and the rest of the values be <0. So now instead of our values ranging from 0 to infinity, we have limited it to 0 and 1