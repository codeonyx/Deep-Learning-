import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


n_inputs=4
n_hidden=3
n_outputs=2


weights_i_h=np.random.normal(0, scale=0.1, size=(n_inputs, n_hidden))
weights_h_o=np.random.normal(0, scale=0.1, size=(n_hidden, n_outputs))

X=np.random.randn(4)

hidden_layer_inputs=np.dot(X, weights_i_h)
hidden_layer_output=sigmoid(hidden_layer_inputs)


print(hidden_layer_output)

output_layer_input=np.dot(hidden_layer_output, weights_h_o)
output_layer_output=sigmoid(output_layer_input)


print(output_layer_output)