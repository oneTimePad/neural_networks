
import numpy as np
import random
import math
import pdb

"""
Fully connected Neural network

based on code from https://github.com/mnielsen/neural-networks-and-deep-learning
"""

"""
other network files are just older versions of this
"""

import regularization_functions as reg
class FCNetwork(object):


    def __init__(self,network_topology,network_activations,cost_fn):
        """
            takes network : [layer1_size,layer2_size,....layer_l_size]
            take activations :[layer2_activation,layer3_activation,...layer_l_activation]
            and intitializes weigts and biases at random
        """

        self.network_topology = network_topology
        self.activations = network_activations
        self.weights = [np.random.randn(net,w)/np.sqrt(w) for net,w in zip(network_topology[1:],network_topology[:-1])]
        self.biases  = [np.random.randn(units,1) for units in network_topology[1:]]
        self.cost = cost_fn


    def compute_mini_batch_gradients(self,mini_batch,drop):
        """
            performs a single weight/bias update for a given mini_batch
        """

        gradient_weights = [np.zeros(w.shape) for w in self.weights]
        gradient_biases  = [np.zeros(b.shape) for b in self.biases]

        for x,y in mini_batch:
            #get gradients for a mini_batch sample
            grad_w,grad_b = self.backprop(x,y,drop)
            #sum gradients over mini_batch
            gradient_weights = [cw+dw for cw,dw in zip(gradient_weights,grad_w)]
            gradient_biases  = [cb+db for cb,db in zip(gradient_biases,grad_b)]
        gradient_weights = [cw/len(mini_batch) for cw in gradient_weights]
        gradient_biases  = [cb/len(mini_batch) for cb in gradient_biases]
        return (gradient_weights,gradient_biases)

    def learn(self,training_data,epochs,mini_batch_size,learning_method,drop_out=False,test_data=None):
        """
            perform the chosen learning method using a random mini_batch for epochs
        """

        training_data = list(training_data)
        n = len(training_data)
        print("Training Samples received: %d"%n)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batchs = [
                training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)
            ]

            for mini_batch in mini_batchs:

                #list of list of indices of all units dropped per layer
                drop = [np.random.randint(0,self.biases[i].shape[0],math.floor(self.biases[i].shape[0]/2)) for i in range(0,len(self.biases)-1)] if drop_out else None

                grad_w,grad_b = self.compute_mini_batch_gradients(mini_batch,drop)
                self.weights,self.biases = learning_method.update((self.weights,grad_w),(self.biases,grad_b))

                if drop_out:
                    #halve all outgoing neurons
                    for i in range(1,len(self.weights)):
                        self.weights[i] = 1/2 *self.weights[i]

            print("Epoch {0}:".format(j),end="")
            if test_data:
                test_data = list(test_data)
                for data in test_data:
                    print(" {0}/{1},".format(self.evaluate(data),len(data)),end="")
            print("\n")


    def evaluate(self,test_data):
        """
        evaluate network performance
        inputs:= inputs to the network
        y:= expected outputs
        """
        #pdb.set_trace()
        test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data ]
        return sum(int(x==y) for (x,y) in test_results)
    def feedforward(self,activation):
        """
        evaluate network in the forward direction
        return the output layer activations
        """
        for w,b,act in zip(self.weights,self.biases,self.activations):
            activation = act.transform(np.dot(w,activation)+b)
        return activation


    def backprop(self,inputs,y,drop):

        """
            peform backpropagation on the fully-connected network
            inputs:= training set sample inputs
            y:= its correct outputs
        """

        activations = [inputs]

        linear_outputs = []
        i = 0
        stop = len(self.weights)-1
        #perform feedforward operation
        for w,b,act in zip(self.weights,self.biases,self.activations):
            linear = np.dot(w,activations[-1])+b
            linear_outputs.append(linear)
            a = act.transform(linear)

            if i != stop and drop:
                a[drop[i]] = 0
            activations.append(a)
            i+=1




        #gradients are the gradient of the cost fct : dC/dz (called delta) where z is the linear output

        #compute the first gradient for the output layer (hadmard product)

        delta_output_layer = self.cost.derivative(activations[-1],y)*self.activations[-1].derivative(linear_outputs[-1])

        #list of dC/dw = dC/dz * dz/dw =dC/dz*activation(prev_layer)
        #matrix of dC/dw = activations[prev_layer] = (#prev_unit,1) times transpose of delta_output_layer =(1,#out_unit)
        gradient_weights = [np.zeros(w.shape) for w in self.weights]
        gradient_weights[-1] = np.dot(delta_output_layer, activations[-2].transpose())
        #list of dC/db=dC/dz
        gradient_biases = [np.zeros(b.shape) for b in self.biases]
        gradient_biases[-1] =  delta_output_layer

        delta = delta_output_layer

        #go back prop through network
        #start at layer 1 --> second layer
        for l in range(2,len(self.network_topology)):
            #back propagate : dC/dZ_L-1 = dC/dZ * dZ/daL *daL*dZL-1 = delta_L *w_L (.) der_act(Z_L-1)
            #transpose of the weight matrix * deltas_for_output_layer (hadmard with) cost_der(z_L-1)
            delta = np.dot(self.weights[-l+1].transpose(),delta)*self.activations[-l].derivative(linear_outputs[-l])

            gradient_weights[-l] = np.dot(delta,activations[-l-1].transpose())
            gradient_biases[-l] = delta

        return (gradient_weights,gradient_biases)
