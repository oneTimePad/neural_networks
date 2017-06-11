
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


    def __init__(self,network_topology,network_activations,cost_fn,drop_out=None,bn=None):
        """
            takes network : [layer1_size,layer2_size,....layer_l_size]
            take activations :[layer2_activation,layer3_activation,...layer_l_activation]
            and intitializes weigts and biases at random
            drop_out:=  probability to drop at that layer
            bn:= [layer2(None/BNLayer),layer3(None/BNLayer),...]
            None/BNLayer(object):= whether to have BatchNorm at the layer (except input/output)
        """
        if len(network_topology)-1 != len(network_activations):
            raise Exception("number of activation functions must match network")

        self.network_topology = network_topology
        self.activations = network_activations
        self.weights = [np.random.randn(net,w)/np.sqrt(w) for net,w in zip(network_topology[1:],network_topology[:-1])]
        self.biases  = [np.random.randn(units,1) for units in network_topology[1:]]
        self.cost = cost_fn
        self.drop_out = drop_out
        self.bn = bn+[None] if bn!=None else [None]*len(self.network_topology)


    def compute_mini_batch_gradients(self,mini_batch):
        """
            performs a single weight/bias update for a given mini_batch
        """

        gradient_weights = [np.zeros(w.shape) for w in self.weights]
        gradient_biases  = [np.zeros(b.shape) for b in self.biases]

        drop = self.drop_out
        for x,y in mini_batch:
            #drop out each unit in all hidden layer
            drop_masks = [ (np.random.randn(*self.biases[i].shape) <(1-drop))/(1-drop) for i in range(0,len(self.biases)-1)] if drop else None

            #add dropout to the input layer
            drop_masks = [(np.random.randn(self.network_topology[0],1) <drop)/drop] +drop_masks if drop else None

            #get gradients for a mini_batch sample
            grad_w,grad_b = self.backprop(x,y,drop_masks)
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
        print("Training Samples received: %d"%n)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batchs = [
                training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)
            ]

            for mini_batch in mini_batchs:
                grad_w,grad_b = self.compute_mini_batch_gradients(mini_batch)
                self.weights,self.biases = learning_method.update((self.weights,grad_w),(self.biases,grad_b))
                #update params of all parametric activation functions
                for act in self.activations:
                    act.update(len(mini_batch))
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
        for w,b,act,bn in zip(self.weights,self.biases,self.activations,self.bn):
            activation = act.transform(np.dot(w,activation)+b if bn is None else bn.transform(np.dot(w,activation),"test"))
        return activation


    def backprop(self,inputs,y,drop_masks):

        """
            peform backpropagation on the fully-connected network
            inputs:= training set sample inputs
            y:= its correct outputs
        """

        activations = [inputs*(drop_masks.pop(0) if drop_masks else 1)]

        linear_outputs = []

        #perform feedforward operation
        for w,b,act,bn in zip(self.weights,self.biases,self.activations,self.bn):
            #batch-norm doesn't used bias (if bn linear outputs get swapped with BN transform output)
            linear = np.dot(w,activations[-1])+b if bn is None else bn.transform(np.dot(w,activations[-1]),"train")
            linear_outputs.append(linear)
            activations.append((act.transform(linear)*(drop_masks.pop(0) if drop_masks and len(drop_masks)!=0 else 1)))


        #gradients are the gradient of the cost fct : dC/dz (called delta) where z is the linear output
        #compute the first gradient for the output layer (hadmard product)
        dL = self.cost.derivative(activations[-1],y)
        #used for activation functions that use backprop
        self.activations[-1].grad_calc(linear_outputs[-1],dL)
        delta_output_layer = dL*self.activations[-1].derivative(linear_outputs[-1])

        #list of dC/dw = dC/dz * dz/dw =dC/dz*activation(prev_layer)
        #matrix of dC/dw = delta outerproduct with activations of prev layer
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
            dL_back = np.dot(self.weights[-l+1].transpose(),delta)
            #for activation functions that require back prop (takes in dL/da)
            self.activations[-l].grad_calc(linear_outputs[-l],dL_back)
            #compuers dL/dz or dL/dy if using bn
            delta = dL_back*self.activations[-l].derivative(linear_outputs[-l])
            #if bn layer, we need to backprop through bn transformation
            #uses dL/dy to compute dL/dz,dL/dg, and dL/dbeta
            delta = delta if self.bn[-l] is None else self.bn[-l].grad_calc(delta)

            gradient_weights[-l] = np.dot(delta,activations[-l-1].transpose())
            gradient_biases[-l] = delta

        return (gradient_weights,gradient_biases)
