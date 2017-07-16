import tensorflow as tf
from functools import partial
import random
import numpy as np

"""
stacked autoencoder with tied weights based off of 
https://github.com/ageron/handson-ml/blob/master/15_autoencoders.ipynb
"""

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150

l2_reg = 0.0001
lr = 0.01

n_epoch = 4
batch_size = 150

he = tf.contrib.layers.variance_scaling_initializer()
reg = tf.contrib.layers.l2_regularizer(l2_reg)
act= tf.nn.elu
X = tf.placeholder(tf.float32, shape=(None,n_inputs),name="X")

def fc_layer(x,n_hidden,scope,bias=None,activation=True,reuse=False,transpose=False,layer_name='hidden'):
    with tf.name_scope(layer_name):
        with tf.variable_scope(scope,reuse=reuse):
                x_size = int(x.get_shape()[1])

                w = tf.get_variable('weights',initializer=he([x_size,n_hidden]))
                w = tf.transpose(w) if transpose else w

        b = tf.Variable(tf.zeros(n_hidden),name='biases') if bias is None else bias
        affine = tf.matmul(x,w)+b
        out = act(affine) if activation else affine
        return out,w,b


with tf.name_scope('network'):
    h1,w1,b1 = fc_layer(X,n_hidden1,'first_stack')
    h2,w2,b2 = fc_layer(h1,n_hidden2,'second_stack')
    h3,_,b3 = fc_layer(h2,n_hidden1,'second_stack',transpose=True,reuse=True)
    out,_,b4 = fc_layer(h3,n_inputs,'first_stack',activation=False,transpose=True,reuse=True,layer_name='out')

aux1,_,__ = fc_layer(h1,n_inputs,'first_stack',bias= b4,activation=False,reuse=True,transpose=True,layer_name='aux1_out')

def aux_net(x,outputs,w,train_vars):

    with tf.name_scope('aux_loss'):
        loss = tf.reduce_mean(tf.square(outputs-x))+reg(w)
    with tf.name_scope('aux_train'):
        opt = tf.train.AdamOptimizer(lr)
        train_op = opt.minimize(loss,var_list=train_vars)
    return train_op
main_loss = tf.reduce_mean(tf.square(out-X))


#using 'network/hidden' pulls all hidden layer biases
train_op1 = aux_net(X,aux1,w1,[tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=s) for s in ['first_stack','network/hidden/biases','network/out']])
train_op2 = aux_net(h1,h3,w2,[tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=s) for s in ['second_stack','network/hidden_1','network/hidden_2']])




init= tf.global_variables_initializer()

from tensorflow.examples.tutorials.mnist import input_data
#change
mnist = input_data.read_data_sets("/home/lie/Downloads/MNIST")


with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('tf_logs/auto_encoder',tf.get_default_graph())
    init.run()
    #phase1
    for epoch in range(n_epoch):
        print("phase1,epoch: %d" %epoch)
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, _ = mnist.train.next_batch(batch_size)
            sess.run(train_op1,feed_dict={X:X_batch})
        loss =sess.run(tf.get_default_graph().get_tensor_by_name('aux_loss/add:0'),feed_dict={X:X_batch})
        print(loss)
    h1_cache = sess.run(h1,feed_dict={X:mnist.train.images})

    #phase2
    for epoch in range(n_epoch):
        print("phase2,epoch: %d" %epoch)
        random.shuffle(h1_cache)
        mini_batches = [
            h1_cache[k:k+batch_size] for k in range(0,np.shape(h1_cache)[0],batch_size)
        ]
        for mini_batch in mini_batches:
            sess.run(train_op2,feed_dict={h1:mini_batch})
        loss =sess.run(tf.get_default_graph().get_tensor_by_name('aux_loss_1/add:0'),feed_dict={h1:mini_batch})
        print(loss)
    """
    for x in mnist.test.images:
        loss =sess.run(main_loss,feed_dict={X:[x]})
        print(loss)
    """
    import matplotlib
    import matplotlib.pyplot as plt
    n_test_digits = 2
    X_test = mnist.test.images[:n_test_digits]
    outputs_val = out.eval(feed_dict={X: X_test})
    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    #savefig('lol.png')
    def plot_image(image, shape=[28, 28]):
        plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
        plt.axis("off")
        plt.show()
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        #plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])
