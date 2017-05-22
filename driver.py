import multi_layer_network
import cost_functions
import activation_functions
import learning_methods
import mnist_loader
import numpy as np
import sys
network_topology = [784,30,10]
network_activations = [activation_functions.Tanh(),activation_functions.Softmax()]
def reduceL(t):
    for index,v in enumerate(t):
        x,y = v
        t[index] = x,np.argmax(y)
    return t

eta =3
epochs = 30
mini_batch=10
net = multi_layer_network.FCNetwork(network_topology, \
                        network_activations, \
                        cost_functions.CrossEntropy()
                        )
train,valid,test = mnist_loader.load_data_wrapper()

train_list = list(train)

net.learn(train_list,epochs,mini_batch,learning_methods.GradientDescent(eta),test_data=[reduceL(train_list[:10000]),list(test),list(valid)])
