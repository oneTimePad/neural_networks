
import numpy as np
import random

import pdb


"""
Fully connected Neural network

based on code from https://github.com/mnielsen/neural-networks-and-deep-learning
"""
class FCNetwork(object):


    def __init__(self,network_size,cost_fn,activation_fn):
        """
            takes network : [later1_size,layer2_size,....layer_l_size]
            and intitializes weigts and biases at random
        """
        self.network_size = network_size
        self.weights = [np.random.randn(net,w) for net,w in zip(network_size[1:],network_size[:-1])]
        self.biases  = [np.random.randn(units,1) for units in network_size[1:]]
        #self.weights = [np.zeros((net,w)) for net,w in zip(network_size[1:],network_size[:-1])]
        #self.biases  = [np.zeros((units,1)) for units in network_size[1:]]
        self.cost = cost_fn
        self.activation = activation_fn

    def compute_mini_batch_gradients(self,mini_batch):
        """
            performs a single weight/bias update for a given mini_batch
        """
        gradient_weights = [np.zeros(w.shape) for w in self.weights]
        gradient_biases  = [np.zeros(b.shape) for b in self.biases]

        for x,y in mini_batch:
            #get gradients for a mini_batch sample
            grad_w,grad_b = self.backprop(x,y)
            #sum gradients over mini_batch
            gradient_weights = [cw+dw for cw,dw in zip(gradient_weights,grad_w)]
            gradient_biases  = [cb+db for cb,db in zip(gradient_biases,grad_b)]
        gradient_weights = [cw/len(mini_batch) for cw in gradient_weights]
        gradient_biases  = [cb/len(mini_batch) for cb in gradient_biases]
        return (gradient_weights,gradient_biases)

    def learn(self,training_data,epochs,mini_batch_size,learning_method,test_data=None):
        """
            perform the chosen learning method using a random mini_batch for epochs
        """



        training_data = list(training_data)
        n = len(training_data)
        print("Training Samples received %d:"%n)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batchs = [
                training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)
            ]

            for mini_batch in mini_batchs:
                grad_w,grad_b = self.compute_mini_batch_gradients(mini_batch)
                self.weights,self.biases = learning_method.update((self.weights,grad_w),(self.biases,grad_b))

            print("Epoch {0}:".format(j),end="")
            if test_data:
                test_data = list(test_data)
                for data in test_data:
                    print(" {0}/{1},".format(self.evaluate(data),len(data)),end="")
            print("\n")

    """
    def evaluate_train(self,train_data):

        test_results = [(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in train_data ]
        return sum(int(x==y) for (x,y) in test_results)
    """
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
        for w,b in zip(self.weights,self.biases):
            activation = self.activation.transform(np.dot(w,activation)+b)
        return activation


    def backprop(self,inputs,y):

        """
            peform backpropagation on the fully-connected network
            inputs:= training set sample inputs
            y:= its correct outputs
        """

        activations = [inputs]

        linear_outputs = []

        #perform feedforward operation
        for w,b in zip(self.weights,self.biases):
            linear = np.dot(w,activations[-1])+b
            linear_outputs.append(linear)
            activations.append(self.activation.transform(linear))

        #gradients are the gradient of the cost fct : dC/dz (called delta) where z is the linear output

        #compute the first gradient for the output layer (hadmard product)

        delta_output_layer = self.cost.derivative(activations[-1],y)*self.activation.derivative(linear_outputs[-1])

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
        for l in range(2,len(self.network_size)):
            #back propagate : dC/dZ_L-1 = dC/dZ * dZ/daL *daL*dZL-1 = delta_L *w_L (.) der_act(Z_L-1)
            #transpose of the weight matrix * deltas_for_output_layer (hadmard with) cost_der(z_L-1)
            delta = np.dot(self.weights[-l+1].transpose(),delta)*self.activation.derivative(linear_outputs[-l])

            gradient_weights[-l] = np.dot(delta,activations[-l-1].transpose())
            gradient_biases[-l] = delta

        return (gradient_weights,gradient_biases)
