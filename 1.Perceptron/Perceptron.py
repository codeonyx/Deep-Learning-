import numpy as np


def sigmoid(x):
    return 1/(1-np.exp(-x))

def sigmoid_d(x):
    return sigmoid(x)*(1-sigmoid(x))
#Input Data
x=np.array([0.1,0.3])
#Target
y=0.2
#Input to the weights
weights=np.array([-0.8, 0.5])
#Learning Rate
learn_rate=0.1
#Linear combination output
h=np.matmul(x, weights)
nn_output=sigmoid(h) 
error=y-nn_output
output_grad=sigmoid_d(h)
error_term=error*output_grad
del_w=[learn_rate*error_term*x[0], learn_rate*error_term*x[1]]


