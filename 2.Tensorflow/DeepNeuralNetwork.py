from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('.', one_hot=True, reshape=False)

import tensorflow as tf

learning_rate=0.01
training_epochs=20
batch_size=128
display_step=1

n_input=784
n_labels=10

n_hidden=256

weights={
        'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'out':tf.Variable(tf.random_normal([n_hidden, n_labels]))
        }
bias={
      'hidden_layer':tf.Variable(tf.random_normal([n_hidden])),
      'out':tf.Variable(tf.random_normal([n_labels]))
      }


x=tf.placeholder('float', [None, 28,28,1])
y=tf.placeholder('float', [None, n_labels])

x_flat=tf.reshape(x, [-1, n_input])

layer_1=tf.add(tf.matmul(x_flat, weights['hidden_layer']), bias['hidden_layer'])
layer_1=tf.nn.relu(layer_1)


logits=tf.add(tf.matmul(layer_1, weights['out']), bias['out'])

#Optimizer
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init=tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y=mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})