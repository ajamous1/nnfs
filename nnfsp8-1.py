import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
							[0.1, 0.5, 0.4],
							[0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]


print(np.mean(-np.log(softmax_outputs[[0,1,2], [class_targets]])))

#implementing loss
#if we were to take the -log of 0, we would get infinity
#to combat this, we need to have a limit, say 1e-7 to 1-1e-7