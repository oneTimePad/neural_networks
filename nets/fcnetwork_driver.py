import fcnetwork

import cost_functions
import activation_functions
import learning_methods
import regularization_functions as reg
import mnist as mnist_loader
import numpy as np
import sys
import batch_norm as bn
network_topology = [784,30,10]
#activation_functions.PReLU(learning_methods.Momentum(.01,.7),20),
#activation_functions.PReLU(learning_methods.Momentum(.01,.7),30)
network_activations = [activation_functions.Tanh(), \
activation_functions.Softmax()]
def reduceL(t):
    for index,v in enumerate(t):
        x,y = v
        t[index] = x,np.argmax(y)
    return t

eta =3
lmbda = 0
epochs = 64
mini_batch=10
net=fcnetwork.FCNetwork(network_topology, \
                        network_activations, \
                        cost_functions.CrossEntropy(),\
                        None, \
                        [bn.BNLayer(learning_methods.Momentum(.01,.7),.06,(30,mini_batch))]\
                        )
train,valid,test = mnist_loader.load_data_wrapper()

train_list = list(train)
#reg.L2Reg(lbmda,len(train_list))
net.learn(train_list,epochs,mini_batch, \
learning_methods.Momentum(eta,.7,reg.L2Reg(lmbda,len(train_list))), \
test_data=[reduceL(train_list[:10000]),list(test),list(valid)])
