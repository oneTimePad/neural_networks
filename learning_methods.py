import numpy as np

class LearningMethod(object):
    """
    template for a NN learning method
    grad_weights:= average changes to weights from a mini_batch
    grad_biases := average changes to biases from a mini_batch
    """
    def update(self,grad_weights,grad_biases):
        raise NotImplementedError


class GradientDescent(LearningMethod):
    def __init__(self,learning_rate):
        self.eta = learning_rate
    def update(self,w,b):
        weights,grad_weights = w
        biases,grad_biases  = b
        eta = self.eta
        #perform gradient descent (current_weight-(learning/N)*grad_weight) ,same for bias
        return ([ w-eta*dw for w,dw in zip(weights,grad_weights)], \
        [ b-eta*db for b,db in zip(biases, grad_biases)])
