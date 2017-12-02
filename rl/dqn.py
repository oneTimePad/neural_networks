import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np


env = gym.make('CartPole-v0')
env.reset()


state = tf.placeholder(shape=(None,4),dtype=tf.float32,name='state')

def tf.variable('estimator_q_network'):
    h1 = slim.fully_connected(state,30,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    h2 = slim.fully_connected(h1,80,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    est_q  = slim.fully_connected(h2,2,activation_fn=None,weights_initializer=tf.contrib.layers.variance_scaling_initializer())

est_q_params = {v.name: v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='estimator_q_network')}

def tf.variable('actor_q_network'):
    h1 = slim.fully_connected(state,30,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    h2 = slim.fully_connected(h1,80,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    act_q  = slim.fully_connected(h2,2,activation_fn=None,weights_initializer=tf.contrib.layers.variance_scaling_initializer())

act_q_params = {v,name: v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='actor_q_network')}

copy_ops = [tf.assign(act_q_params[k],est_q_params[k]) for k in est_q_params.keys()]

#batch of sarsa memories
memory_batch = tf.placeholder(shape(None,5),dtype=tf.float32,name='memory_batch')
