import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


"""
variational autoencoder with based off of
https://github.com/ageron/handson-ml/blob/master/15_autoencoders.ipynb
"""

fc = slim.fully_connected
n_epoch = 60
n_inputs = 28 * 28
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20
n_hidden4 = n_hidden2
n_hidden5 = n_hidden3
n_outputs = n_inputs

batch_size = 150

lr = 0.001
"""
with slim.arg_scope(
        [fc],
        activation_fn = tf.nn.elu,
        weights_initializer = tf.contrib.layers.variance_scaling_initializer()
):
    with tf.name_scope("network"):
        X = tf.placeholder(tf.float32, shape=(None,n_inputs),name="X")
        #h2 = slim.stack(X,fc,[n_hidden1,n_hidden2])
        h1 = fc(X,n_hidden1)
        h2 = fc(h1,n_hidden2)
        h3_mean = fc(h2,n_hidden3,activation_fn=None)
        h3_gamma = fc(h2,n_hidden3,activation_fn=None)
        h3_sigma = tf.exp(0.5 * h3_gamma)

        noise = tf.random_normal(tf.shape(h3_sigma),dtype=tf.float32)

        h3 = h3_mean + h3_sigma * noise

        #h5 = slim.stack(h3,fc,[n_hidden4,n_hidden5])
        h4 = fc(h3,n_hidden4)
        h5 = fc(h4,n_hidden5)
        out = fc(h5,n_outputs,activation_fn=None,scope="out")
"""

from functools import partial
initializer = tf.contrib.layers.variance_scaling_initializer()
my_dense_layer = partial(
    tf.layers.dense,
    activation=tf.nn.elu,
    kernel_initializer=initializer)


X = tf.placeholder(tf.float32, [None, n_inputs])
hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
h3_mean = my_dense_layer(hidden2, n_hidden3, activation=None)
h3_gamma = my_dense_layer(hidden2, n_hidden3, activation=None)
noise = tf.random_normal(tf.shape(h3_gamma), dtype=tf.float32)
h3 = h3_mean + tf.exp(0.5 * h3_gamma) * noise
hidden4 = my_dense_layer(h3, n_hidden4)
hidden5 = my_dense_layer(hidden4, n_hidden5)
out = my_dense_layer(hidden5, n_outputs, activation=None)

with tf.name_scope("loss"):
    """
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=X,logits=out))
    latent_loss = 0.5 * tf.reduce_sum(tf.exp(h3_gamma) + tf.square(h3_mean)-1 -h3_gamma)
    loss = loss + latent_loss
    """
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=out)
    reconstruction_loss = tf.reduce_sum(xentropy)
    latent_loss = 0.5 * tf.reduce_sum(
    tf.exp(h3_gamma) + tf.square(h3_mean) - 1 - h3_gamma)
    loss = reconstruction_loss + latent_loss

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss)

from tensorflow.examples.tutorials.mnist import input_data
#change
mnist = input_data.read_data_sets("/home/lie/Downloads/MNIST")


init = tf.global_variables_initializer()

n_digits= 60
with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('tf_logs/auto_encoder',tf.get_default_graph())
    init.run()
    #phase1
    for epoch in range(n_epoch):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, _ = mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={X:X_batch})
        test_loss =sess.run(loss,feed_dict={X:X_batch})
        print(test_loss)
    codings_rnd = np.random.normal(size=[n_digits,n_hidden3])
    sig = tf.nn.sigmoid(out)
    outputs_val = sig.eval(feed_dict={h3:codings_rnd})

import matplotlib
import matplotlib.pyplot as plt

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")
    plt.show()
for iteration in range(n_digits):
    #plt.subplot(n_digits,10,iteration+1)
    plot_image(outputs_val[iteration])
