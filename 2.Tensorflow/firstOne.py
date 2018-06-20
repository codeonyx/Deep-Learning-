import tensorflow as tf

#Creating the methods

def get_weights(n_features, n_labels):
    #Here, we are defining a method to generate the weights matrix 
    return tf.Variable(tf.truncated_normal((n_features,n_labels)))

def get_biases(n_labels):
    #Here, we are defining a method to generate the bias matrix
    return tf.Variable(tf.zeros(n_labels))

def linear(input, w, b):
    #Here, we are performing the linear operation y=Wx+b 
    return tf.add(tf.matmul(input,w), b)

#Importing the mnist dataset
from tensorflow.examples.tutorials.mnist import input_data


def mnist_features_labels(n_labels):
#Here, we get the first n labels from the MNIST dataset 
    mnist_features=[]
    mnist_labels=[]
    
    mnist=input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)
    
    for mnist_feature, mnist_label in zip(*mnist.train.next_batch(10000)):

        # Add features and labels if it's for the first <n>th labels
        if mnist_label[:n_labels].any():
            mnist_features.append(mnist_feature)
            mnist_labels.append(mnist_label[:n_labels])

    return mnist_features, mnist_labels

#Features and Labels
features=tf.placeholder(tf.float32)
labels=tf.placeholder(tf.float32)

#Define the number of features and the number of labels
n_features=784
n_labels=3

#Defining the weights and bias
w = get_weights(n_features, n_labels)
b= get_biases(n_labels)

#Linear Function WX+b
logits=linear(features, w, b)

train_features, train_labels=mnist_features_labels(n_labels)

with tf.Session() as session:
    session.run(tf.global_variables_initializer()) 
    prediction=tf.nn.softmax(logits)
    cross_entropy=-tf.reduce_sum(labels*tf.log(prediction), reduction_indices=1)
    loss=tf.reduce_mean(cross_entropy)   
    learning_rate=0.08
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    #Run the optimizer and get the loss
    _, l=session.run([optimizer, loss], feed_dict={features: train_features, labels:train_labels})


print('Loss: {}'.format(1))
    



        


    