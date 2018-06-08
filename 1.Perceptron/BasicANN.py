import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

x=np.array([0.1, 0.3])

y=0.2
learn_rate=0.1

weights=np.array([-0.8,0.5])

h=np.matmul(x, weights)

yhat=sigmoid(h)

error=y-yhat

output_gradient = sigmoid_prime(yhat)
error_term=error*output_gradient

del_w=learn_rate*error_term*x