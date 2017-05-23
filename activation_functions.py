
import numpy as np
import pdb
class ActivationFunction(object):
    """
    template for network activation function
    unit_outputs:= the linear outputs of a given unit
    """
    def transform(self,unit_outputs):
        raise NotImplementedError
    def derivative(self,unit_outputs):
        raise NotImplementedError
    def update(self,half_deltas):
        """
        used by activation functions that require backprop
        """
        raise NotImplementedError

class Softmax(ActivationFunction):
    """
    typically used for output layer
    """
    def transform(self,unit_outputs):
        cpy = np.copy(unit_outputs)
        num = np.exp(cpy)
        den = sum(num)
        return num/den
    def derivative(self,unit_outputs):
        return self.transform(unit_outputs)*(1-self.transform(unit_outputs))
    def update(self,unit_outputs,half_deltas):
        pass



class Sigmoid(ActivationFunction):
    """
    can be used for both hidden and output layer (better for output)
    """
    def transform(self,unit_outputs):

        return 1.0/(1.0+np.exp(-np.copy(unit_outputs)))
    def derivative(self,unit_outputs):
        return self.transform(unit_outputs)*(1-self.transform(unit_outputs))
    def update(self,unit_outputs,half_deltas):
        pass

class Tanh(ActivationFunction):
    """
    only use as hidden
    """
    def transform(self,unit_outputs):
        return np.tanh(unit_outputs)
    def derivative(self,unit_outputs):
        return 1.0 - np.tanh(unit_outputs)**2
    def update(self,unit_outputs,half_deltas):
        pass

class ReLU(ActivationFunction):
    """
    only use for hidden
    """
    def transform(self, unit_outputs):
        """
        zeros = np.zeros(unit_outputs.shape)
        return np.maximum.reduce([unit_outputs,zeros])
        """
        cpy = np.copy(unit_outputs)
        return np.maximum(cpy,0,cpy)
    def derivative(self,unit_outputs):
        der= np.zeros(unit_outputs.shape)
        ones = unit_outputs >=0
        der[ones] = 1
        return der
    def update(self,unit_outputs,half_deltas):
        pass
class LeakyReLU(ActivationFunction):
    """
    only use for hidden
    """
    def transform(self, unit_outputs):
        zeros = np.zeros(unit_outputs.shape)
        ones =  np.ones(unit_outputs.shape)*.01
        greater_than_zero = np.maximum.reduce([unit_outputs,zeros])
        less_than_zero= np.minimum.reduce([unit_outputs,zeros])*ones
        return (greater_than_zero+less_than_zero)
    def derivative(self,unit_outputs):
        der = np.zeros(unit_outputs.shape)
        ones = unit_outputs >0
        der[ones] = 1
        zeros = unit_outputs <=0
        der[zeros] = .01
        return der
    def update(self,unit_outputs,half_deltas):
        pass

class PReLU(ActivationFunction):
    """
    only used for hidden, parameterized by slope in negative region
    PReLU From paper :
    "Delving Deep into Rectifiers:Surpassing Human-Level
    Performance on ImageNet Classification: https://arxiv.org/pdf/1502.01852.pdf  "
    """
    def __init__(self,learning_method,layer_size):
        self.alpha = None
        self.size = layer_size
        self.learning_method = learning_method
    def transform(self,unit_outputs):
        if self.alpha is None:
            self.alpha = np.random.randn(self.size,1)
        zeros = np.zeros(unit_outputs.shape)
        ones =  np.ones(unit_outputs.shape)*self.alpha
        greater_than_zero = np.maximum.reduce([unit_outputs,zeros])
        less_than_zero= np.minimum.reduce([unit_outputs,zeros])*ones
        return (greater_than_zero+less_than_zero)
    def derivative(self,unit_outputs):
        der = np.zeros(unit_outputs.shape)
        ones = unit_outputs >0
        der[ones] = 1
        zeros = unit_outputs <=0
        der[zeros] = self.alpha[zeros]
        return der
    def update(self,unit_outputs,half_deltas):
        #computes dact/dalpha (this might not make a difference, since alpha for >0 are ignored)
        cpy = np.copy(unit_outputs)
        zeros = unit_outputs >0
        cpy[zeros] = 0
        #computes dL/dalpha
        grad_alpha = cpy * half_deltas
        self.alpha, _ = self.learning_method.update(([self.alpha],[grad_alpha]),([],[]))
        self.alpha = self.alpha[0]
