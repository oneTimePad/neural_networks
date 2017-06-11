import tensorflow as tf
import random
import numpy as np
from datetime import datetime
"""
Simple implementation of an FC network using Tensorflow
"""


def numpy_to_tensorflow_training(mini_batch):
    """
    training data comes in as [(x,y),(x,y),...]
    outputs [[x],[x],[x],...],[[y],[y],[y],..]
    ravels training data and puts it in 1D array form that
    tensorflow optimizer likes (for simulaneous batch training)
    """
    unzip = list(zip(*mini_batch))
    x = list(unzip[0])
    y = list(unzip[1])
    x = [t.ravel() for t in x]
    y = [t.ravel() for t in y]
    return x,y

def affine(X,layer_size):
    with tf.name_scope("Affine"):
        input_size =X.get_shape().as_list()[1]
        weights = tf.Variable(tf.div(tf.random_normal((input_size,layer_size)),np.sqrt(input_size)),name="weights")
        biases  = tf.Variable(tf.random_normal((1,layer_size)),name="biases")

        val = tf.add(tf.matmul(X,weights),biases,name="z")
        return val

def relu(X):
    with tf.name_scope("ReLU"):
        return tf.maximum(X,0)

def sigmoid(X):
    with tf.name_scope("Sigmoid"):
        return tf.sigmoid(X)
def softmax(X):
    with tf.name_scope("Softmax"):
        return tf.nn.softmax(X)

def cross_entropy(X,y):
    with tf.name_scope("CrossEntropy"):
        return tf.reduce_mean(-tf.reduce_sum(y * tf.log(X), reduction_indices=[1]))

def quadratic_cost(X,y):
    with tf.name_scope("QuadraticCost"):
        return tf.reduce_mean(tf.square(y-X),name="cost")

class FCNetwork(object):

    def __init__(self,network_topology,activations,cost,rate):
        self.net = network_topology
        self.input = tf.placeholder(tf.float32,shape=(None,network_topology[0]),name="X")
        self.ground_truth = tf.placeholder(tf.float32,shape=(None,network_topology[-1]),name="y")
        self.graph = self.input
        for act,size in zip(activations,network_topology[1:]):
            self.graph=affine(self.graph,size)
            if act is not None:
                self.graph = act(self.graph)

        self.cost = cost(self.graph,self.ground_truth)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate)
        #from  https://stackoverflow.com/questions/36498127/how-to-effectively-apply-gradient-clipping-in-tensor-flow/36501922
        gvs = optimizer.compute_gradients(self.cost)
        capped_gvs = [(tf.clip_by_value(grad,-1.,1.),var) for grad, var in gvs]
        self.training_opt = optimizer.apply_gradients(capped_gvs)
        #self.training_opt = optimizer.minimize(self.cost)

    def train(self,training_data,epochs,mini_batch_size,test_data=None):

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            root_logdir = "tf_logs"
            logdir = "{}/run-{}".format(root_logdir,now)
            file_writer =tf.summary.FileWriter(logdir,tf.get_default_graph())
            cost_sum = tf.summary.scalar('Cost',self.cost)
            init.run()
            training_data= list(training_data)
            n = len(training_data)
            for j in range(epochs):
                random.shuffle(training_data)
                mini_batchs = [
                    training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)
                ]
                for mini_batch in mini_batchs:
                    x,y = numpy_to_tensorflow_training(mini_batch)

                    sess.run(self.training_opt,feed_dict={self.input:x,self.ground_truth:y})


                if test_data:
                    num_correct =0
                    test_data = list(test_data)
                    last_data = None
                    for data in test_data:
                        #formats test data into correct format for tensor optimizing
                        #y is represented as a single int it is converted to one-hot representation
                        x,y = data
                        x =  x.reshape((1,-1))
                        y_m =  np.zeros((self.net[-1],1))
                        y_m[y]=1
                        y_m=y_m.reshape((1,-1))
                        out = sess.run(self.graph,feed_dict=({self.input:x,self.ground_truth:y_m}))
                        num_correct+=(np.argmax(out)==y)
                        last_data = (x,y_m)
                    print("%d/%d"%(num_correct,len(test_data)))
                    x,y = last_data
                    summary = cost_sum.eval(feed_dict={self.input:x,self.ground_truth:y})
                    file_writer.add_summary(summary,j)
