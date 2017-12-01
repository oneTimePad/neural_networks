import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np


env = gym.make('CartPole-v0')
env.reset()


#estimate action-value function
state_action = tf.placeholder(shape=(None,5),dtype=tf.float32,name='state_action')
with tf.variable_scope('critic'):
    h1 = slim.fully_connected(state_action,512,activation_fn=tf.nn.relu)
    h2 = slim.fully_connected(h1,256,activation_fn=tf.nn.relu)
    Q = slim.fully_connected(h2,1,activation_fn=None)

#state_ph = tf.placeholder(shape=(None,4),dtype=tf.float32)
with tf.variable_scope('actor'):
    h1 = slim.fully_connected(state_action,512,activation_fn=tf.nn.relu)
    h2 = slim.fully_connected(h1,256,activation_fn=tf.nn.relu)
    logits = slim.fully_connected(h2,1,activation_fn=None)
    action_fn = tf.nn.sigmoid(logits)

critic_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='critic')
actor_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='actor')

alpha = .9
beta = .95
gamma = .3
delta_ph = tf.placeholder(shape=(None,1),dtype=tf.float32,name='delta_ph')


actor_update =[ tf.assign(v,v+tf.clip_by_value(tf.scalar_mul(alpha,tf.gradients(tf.log(tf.squeeze(action_fn)),v)[0])*tf.squeeze(Q),-.5,.5)) for v in actor_param]
critic_update  =[ tf.assign(v,v+tf.clip_by_value(beta*tf.squeeze(delta_ph)*tf.gradients(tf.squeeze(Q),v)[0],-1,1)) for v in critic_param]
init = tf.global_variables_initializer()

def left_prob(state,sess):
    return sess.run(action_fn,feed_dict={'state_action:0':[[*state,0]]})
def right_prob(state,sess):
    return sess.run(action_fn,feed_dict={'state_action:0':[[*state,1]]})
def get_action(sess,state):
    probs = [left_prob(state,sess),right_prob(state,sess)]
    #print(probs)
    return np.argmax(probs)
with tf.Session() as sess:
    init.run()
    for episode in range(1,1001):
        done = False,
        G,reward = 0,0
        state = env.reset()
        action = get_action(sess,state)
        steps = 0
        while done!=True:
            prev_state = state
            state, reward,done,info = env.step(action)
            env.render()
            prev_action = action
            action = get_action(sess,state)
            print(action)
            #TD-delta computation
            sa = np.concatenate([state,[float(action)]])
            prev_sa = np.concatenate([prev_state,[float(prev_action)]])
            Q_next = sess.run(Q,feed_dict={'state_action:0':[sa]})
            Q_prev = sess.run(Q,feed_dict={'state_action:0':[prev_sa]})
            delta = reward+gamma*Q_next - Q_prev
            sess.run(actor_update,feed_dict={'state_action:0':[prev_sa]})
            sess.run(critic_update,feed_dict={'delta_ph:0':delta,'state_action:0':[prev_sa]})
            #print(sess.run(actor_param))
            steps+=1
        print('FAILED %d'%steps)
