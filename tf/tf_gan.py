import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
n_epochs = 100000
sampl= 1000
code = 100
bsize = 128

n_hidden = 128
ng_output=28*28
nd_output=1
lr = 0.01

Z = tf.placeholder(tf.float32,shape=(None,code),name='z')
X = tf.placeholder(tf.float32,shape=(None,ng_output),name='x')

g_layers =  [(n_hidden,tf.nn.relu),(ng_output,None)]
d_layers =  [(n_hidden,tf.nn.relu),(nd_output,None)]

#gen net
g = tf.nn.sigmoid(slim.stack(Z,slim.fully_connected,g_layers,scope='gen'))

#dis net on data
dx = slim.stack(X,slim.fully_connected,d_layers,scope='dis')
#dis net on gen data
dg = slim.stack(g,slim.fully_connected,d_layers,scope='dis',reuse=True)

with tf.name_scope('dis_train'):
    opt = tf.train.AdamOptimizer()
    #dloss = -tf.reduce_mean(tf.log(dx+e)+tf.log(1-dg+e))
    dloss_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dx,labels=tf.ones_like(dx)))
    dloss_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dg,labels=tf.zeros_like(dg)))
    dloss = dloss_r+dloss_f

    dtrain_op = opt.minimize(dloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='dis'))
with tf.name_scope('gen_train'):
    opt = tf.train.AdamOptimizer()
    #gloss = -tf.reduce_mean(tf.log(dg+e))
    gloss =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dg,labels=tf.ones_like(dg)))
    gtrain_op = opt.minimize(gloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='gen'))


"""
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, 784])

D_W1 = tf.Variable(xavier_init([784, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


Z = tf.placeholder(tf.float32, shape=[None, 100])

G_W1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, 784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

with tf.name_scope('dis_train'):
    opt = tf.train.AdamOptimizer()
    #dloss = -tf.reduce_mean(tf.log(dx+e)+tf.log(1-dg+e))
    dloss_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,labels=tf.ones_like(D_logit_real)))
    dloss_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,labels=tf.zeros_like(D_logit_fake)))
    dloss = dloss_r+dloss_f

    dtrain_op = opt.minimize(dloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='dis'))
with tf.name_scope('gen_train'):
    opt = tf.train.AdamOptimizer()
    #gloss = -tf.reduce_mean(tf.log(dg+e))
    gloss =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dg,labels=tf.ones_like(dg)))
    gtrain_op = opt.minimize(gloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='gen'))


D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
dloss = D_loss_real + D_loss_fake
gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

dtrain_op = tf.train.AdamOptimizer().minimize(dloss, var_list=theta_D)
gtrain_op = tf.train.AdamOptimizer().minimize(gloss, var_list=theta_G)
"""

init= tf.global_variables_initializer()


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

if not os.path.exists('gan/'):
    os.makedirs('gan/')

def sample(batch_size,code):
    return np.random.uniform(-1.,1.,size=[batch_size,code])

mnist = input_data.read_data_sets("/home/lie/Downloads/MNIST")

#change
i = 0
mbatch_size =16
with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('tf_logs/gan',tf.get_default_graph())
    init.run()
    for epoch in range(n_epochs):
        if epoch % sampl == 0:
            samples = sess.run(g,feed_dict={Z:sample(16,code)})

            fig = plot(samples)
            plt.savefig('gan/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        for __ in range (10):
            X_batch, __ = mnist.train.next_batch(bsize)
            _,dloss_val= sess.run([dtrain_op,dloss],feed_dict={X:X_batch,Z:sample(bsize,code)})
        X_batch, __ = mnist.train.next_batch(bsize)
        _,gloss_val= sess.run([gtrain_op,gloss],feed_dict={Z:sample(bsize,code)})

        if epoch%1000 ==0:
            print("DLoss %f" %dloss_val)
            print("Gloss %f" %gloss_val)
