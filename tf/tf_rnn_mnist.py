import tensorflow as tf

"""
Recurrent NN for MNIST class
from 'Hands-On Machine Learning with Scikit-Learn and Tensorflow'
"""

MNIST_LOCATION='~/Downloads/MNIST'
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

lr = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32,[None])

with tf.name_scope('RNN'):
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell,X,dtype=tf.float32)
with tf.name_scope('output'):
    he_init = tf.contrib.layers.variance_scaling_initializer()
    logits = tf.layers.dense(
                    states,
                    kernel_initializer=he_init,
                    units = n_outputs,
                    name='logits'
    )
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=y
    )
    loss = tf.reduce_mean(xentropy)
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(MNIST_LOCATION)
X_test = mnist.test.images.reshape([-1,n_steps,n_inputs])
y_test = mnist.test.labels

n_epochs = 100
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch,y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1,n_steps,n_inputs))
            sess.run(train_op,feed_dict={X:X_batch,y:y_batch})
        acc_train = accuracy.eval(feed_dict={X:X_batch,y:y_batch})
        acc_test  = accuracy.eval(feed_dict={X:X_test,y:y_test})
        print(epoch,"Train accuracy:",acc_train,"Test accuracy:",acc_test)
