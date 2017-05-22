import numpy as np
class CostFunction(object):
    """
    template for network cost function

    activations := network activations at the output layer
    y:= the correct answers
    """

    def cost(self,activations,y):
        raise NotImplementedError
    def derivative(self,activations,y):
        raise NotImplementedError

class QuadraticCost(CostFunction):
    def cost(self,activations,y):
        return (1/(2*len(y)))*sum((activations-y)^2)
    def derivative(self,activations,y):
        return (activations-y)

class CrossEntropy(CostFunction):
    def cost(self,activations,y):
        pass #never used in backprop
    def derivative(self,activations,y):
        return (activations-y)/(activations*(1-activations))
