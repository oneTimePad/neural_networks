import numpy as np

"""
Implements the batch normalization layer
specified in this paper: Batch Normalization: Accelerating Deep Network Training by
Reducing Internal Covariate Shift

References: https://github.com/martinkersner/cs231n/blob/master/assignment2/layers.py
"""


class BNLayer(object):

    def __init__(self,learning_method,momentum,m,esp=.001,layer_size):
        self.gamma = np.random.rand(layer_size,1)
        self.beta  = np.random.rand(layer_size,1)
        self.running_mean =0
        self.running_var  =1
        self.eps = eps
        #contains activations for each mini_batch sample
        self.last_sample.np.zeros(layer_size,1)

    def transform(self,unit_outputs,mode):
        if mode == "train":
            #compute the sample mean/var
            sample_mean = np.mean(unit_outputs,axis=0)
            sample_var  = np.mean(unit_outputs,axis=0)
            #normalize the input
            u_hat = (unit_outputs-sample_mean)/np.sqrt(sample_var+self.eps)
            #update the running mean/var
            self.running_mean = self.momentum*self.running_mean+(1-self.momentum)*sample_mean
            self.running_var  = self.momentum*self.running_var +(1-self.momentum)*sample_var

            self.last_sample = u_hat
            #apply affine transform
            return u_hat*self.gamma+self.beta
        elif mode == "test":
            #use population statistics
            u_hat = (unit_outputs-self.running_mean)/np.sqrt(self.running_var+self.eps)
            return u_hat*self.gamma+self.beta
    def derivative(self):
        #affine transform, derivative is just coef
        return self.gamma
    def update(self,unit_outputs,delta):
        """
        Linear_unit(z)->BN(y)->Non-Linearity(a)
        dL/dy = delta= (dL/a*da/y)
        where y=gamma*x_hat+beta
        """
        #dL/dx
        dx = self.derivative()*delta
        #dL/bg
        dg = delta*self.last_sample
        #dL/db
        db = delta
