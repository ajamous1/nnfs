#it's not only important for us to know whether our network is right or wrong
#we must also know HOW right or wrong it is, so that we can train it
#mean absolute error: we just take the average of the distances from the actual vs expected
#the closer to the desired value the closer to the actual desired function 
#in general, we use categorical cross-entropy
#which is taking the negative sum of the target value, multiplied by the log of the predicted value, for each of the values in the distribution
#it can be simplified using one hot encoding to simply the negative log
#in programming with AI, we generally assume log = ln

import math
math.log

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] +
		 math.log(softmax_output[1])*target_output[1] +
		 math.log(softmax_output[2])*target_output[2])

#where the confidence is higher, loss is lower, where the confidence is lower, loss is higher, which makes sense

print(loss)
