# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:41:06 2018

@author: s.shankar.bakale
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data', validation_size=0)

img=mnist.train.images[2]
plt.imshow(img.reshape((28,28)), cmap='gray')

#Size of the encoding layer(The hidden layer)

encoding_dim=32
image_size=mnist.train.images.shape[1]

inputs=tf.placeholder(tf.float32, shape=(None, image_size), name='inputs')
targets=tf.placeholder(tf.float32, shape=(None,image_size), name='targets')

#Output of the hidden layer

encoded=tf.layers.dense(inputs, encoding_dim, activation=tf.nn.relu)

#Output layer logits

logits=tf.layers.dense(encoded, image_size,activation=None)

#Sigmoid output from 
decoded=tf.nn.sigmoid(logits,name='output')

loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)

cost=tf.reduce_mean(loss)
optimizer=tf.train.AdamOptimizer(0.01).minimize(cost)



#Training
sess=tf.Session()


epochs=20
batch_size=200
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(mnist.train.num_examples//batch_size):
        batch=mnist.train.next_batch(batch_size)
        feed={inputs:batch[0], targets:batch[0]}
        batch_cost,_=sess.run([cost, optimizer], feed_dict=feed)
        print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.4f}".format(batch_cost))
        
        
#Checking out the results

fig, axes=plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
in_imgs=mnist.test.images[:10]
reconstructed, compressed=sess.run([decoded, encoded], feed_dict={inputs:in_imgs})

for images, row in zip([in_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)


sess.close()