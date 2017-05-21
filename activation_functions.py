
import numpy as np
class ActivationFunction(object):
    """
    template for network activation function
    unit_outputs:= the linear outputs of a given unit
    """
    def transform(self,unit_outputs):
        raise NotImplementedError
    def derivative(self,unit_outputs):
        raise NotImplementedError

class Sigmoid(ActivationFunction):
    def transform(self,unit_outputs):

        return 1.0/(1.0+np.exp(-unit_outputs))
    def derivative(self,unit_outputs):

        return self.transform(unit_outputs)*(1-self.transform(unit_outputs))

class Tanh(ActivationFunction):
    def transform(self,unit_outputs):
        return np.tanh(unit_outputs)
    def derivative(self,unit_outputs):
        return 1.0 - np.tanh(unit_outputs)**2


class ReLU(ActivationFunction):
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

class LeakyReLU(ActivationFunction):
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
