

import tensorflow as tf
import numpy as np
import gzip
"""
reference: Hands-On Machine Learning with Scikit-Learn and Tensorflow
"""


n_inputs = 784
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None,n_inputs),name="X")
y = tf.placeholder(tf.int64,shape=(None),name="y")

def neuron_layer(X,n_neurons,name,activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        #weight initialization
        stddev = 2/np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)
        w = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]),name="biases")
        z = tf.matmul(X,w)+b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

use = "2"
logits = None
with tf.name_scope("dnn"):
    if use =="1":
        hidden1 = neuron_layer(X,n_hidden1,"hidden1",activation="relu")
        hidden2 = neuron_layer(hidden1,n_hidden2,"hidden2",activation="relu")
        logits = neuron_layer(hidden2,n_outputs,"outputs")
    else:
        from tensorflow.contrib.layers import fully_connected
        hidden1 =fully_connected(X,n_hidden1,scope="hidden1")
        hidden2 =fully_connected(hidden1,n_hidden2,scope="hidden2")
        logits = fully_connected(hidden2,n_outputs,scope="outputs",\
                                activation_fn=None)
loss = None
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(\
                labels=y,logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")

training_op = None
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

accuracy = None
with tf.name_scope("eval"):
    #in output look the high probability
    correct = tf.nn.in_top_k(logits,y,1)
    #convert to float and take the avg
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/home/lie/Downloads/MNIST')

n_epochs = 400
batch_size = 50
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch,y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        acc_train = accuracy.eval(feed_dict={X:X_batch,y:y_batch})
        acc_test = accuracy.eval(feed_dict={X:mnist.test.images,y:mnist.test.labels})
        print(epoch, "Train accuracy:",acc_train,"Test accuracy:",acc_test)
    save_path = saver.save(sess,"./model.ckpt")
