import numpy as np

"""
Implements the batch normalization layer
specified in this paper: Batch Normalization: Accelerating Deep Network Training by
Reducing Internal Covariate Shift

References: https://github.com/martinkersner/cs231n/blob/master/assignment2/layers.py
"""


class BNLayer(object):

    def __init__(self,learning_method,momentum,size,eps=.001):
        """
        learning_method:= method used to learn parameters (gamma,beta)
        momentum:= not to be confused with Momentum Learning, this is used
        for the running mean and running variance
        size:= (layer_size[num_units],batch_size)
        esp:= parameter used when dividing by the std_dev(sqrt(variance)), prevents div by 0
        """
        layer_size, batch_size = size
        self.batch_size = batch_size
        self.gamma = np.random.rand(layer_size,1)
        self.beta  = np.random.rand(layer_size,1)
        self.running_mean =np.zeros((layer_size,1))
        self.running_var  =np.ones((layer_size,1))
        self.eps = eps
        self.momentum = momentum
        #contains last forward pass values for this layer
        self.last_sample = None
        self.last_sample_mean =None
        self.last_sample_var  = None
        self.last_sample_hat = None
        #cumulative gradients for various parameters
        self.grad_gamma = np.zeros((layer_size,1))
        self.grad_beta  = np.zeros((layer_size,1))
        self.grad_sample_mean = np.zeros((layer_size,1))
        self.grad_sample_var  = np.zeros((layer_size,1))

        self.learning_method = learning_method

    def transform(self,unit_outputs,mode):
        if mode == "train":
            #use sample statistics
            #compute the sample mean/var
            sample_mean = np.mean(unit_outputs,axis=0)
            sample_var  = np.var(unit_outputs,axis=0)
            #normalize the input
            u_hat = (unit_outputs-sample_mean)/np.sqrt(sample_var+self.eps)
            #update the running mean/var
            self.running_mean = self.momentum*self.running_mean+(1-self.momentum)*sample_mean
            self.running_var  = self.momentum*self.running_var +(1-self.momentum)*sample_var

            self.last_sample_hat = np.copy(u_hat)
            self.last_sample = np.copy(unit_outputs)
            self.last_sample_mean = sample_mean
            self.last_sample_var = sample_var
            #apply affine transform
            return u_hat*self.gamma+self.beta
        elif mode == "test":
            #use population statistics
            u_hat = (unit_outputs-self.running_mean)/np.sqrt(self.running_var+self.eps)
            return u_hat*self.gamma+self.beta
    def derivative(self):
        #affine transform, derivative is just coef
        return self.gamma
    def grad_calc(self,delta):
        """
        Linear_unit(z)->BN(y)->Non-Linearity(a)
        dL/dy = delta= (dL/a*da/y)
        where y=gamma*x_hat+beta
        """
        #dL/bg (accumulate)
        self.grad_gamma+=delta*self.last_sample_hat
        #dL/db
        self.grad_beta+=delta
        #dL/dx_hat
        dx_hat = self.derivative()*delta
        dvar= dx_hat*(self.last_sample-self.last_sample_mean)*(-1/2)*(self.last_sample_var+self.eps)**(-1.5)
        self.grad_sample_var += dvar
        dmean=(dx_hat*-1/np.sqrt(self.last_sample_var+self.eps))+dvar*-2*(self.last_sample-self.last_sample_mean)/self.batch_size
        self.grad_sample_mean+=dmean
        dx=dx_hat*(1/np.sqrt(self.last_sample_var+self.eps))+dvar*2*(self.last_sample-self.last_sample_mean)/self.batch_size +dmean/self.batch_size
        return dx
    def update(self):
        """
        perform update of parameters gamma and beta
        """
        self.gamma,self.beta = self.learning_method(([self.gamma],[self.grad_gamma/self.batch_size]),(self.beta,self.grad_beta/self.batch_size))
