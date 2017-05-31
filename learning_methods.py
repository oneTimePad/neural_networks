import numpy as np
import regularization_functions as reg


class LearningMethod(object):
    """
    template for a NN learning method
    grad_weights:= average changes to weights from a mini_batch
    grad_biases := average changes to biases from a mini_batch
    """
    def update(self,grad_weights,grad_biases):
        raise NotImplementedError


class AdaGrad(LearningMethod):
    """
    implements adagrad update
    """
    def __init__(self,learning_rate,reg=reg.Regularization()):
        self.eta = learning_rate
        self.reg = reg
        self.mem_w = None
        self.mem_b = None
    def update(self,w,b):
        weights,grad_weights = w
        biases,grad_biases  = b

        if self.mem_w is None:
            self.mem_w = [ np.zeros_like(w) for w in weights]
        if self.mem_b is None:
            self.mem_b =  [ np.zeros_like(b) for b in biases ]

        self.mem_w = [mem+d**2 for mem,d in zip(self.mem_w,grad_weights)]
        self.mem_b = [mem+d**2 for mem,d in zip(self.mem_b,grad_biases)]

        return [ w-self.eta*d/np.sqrt(mem+1e-8) for w,d,mem in zip(weights,grad_weights,self.mem_w)], \
        [ b-self.eta*d/np.sqrt(mem+1e-8) for b,d,mem in zip(biases,grad_biases,self.mem_b)],



class GradientDescent(LearningMethod):
    """
    implements the gradient descent update
    w -> w-eta*gradC_w
    b -> b-eta*gradC_b
    """
    def __init__(self,learning_rate,reg=reg.Regularization()):
        self.eta = learning_rate
        self.reg = reg
    def update(self,w,b):
        weights,grad_weights = w
        biases,grad_biases  = b
        eta = self.eta
        #perform gradient descent (current_weight-(learning/N)*grad_weight) ,same for bias
        return ([ w-eta*dw-(eta*self.reg.derivative(w)) for w,dw in zip(weights,grad_weights)], \
        [ b-eta*db for b,db in zip(biases, grad_biases)])


class Momentum(LearningMethod):
    """
    implements the Momentum update
    v->mu*v-gradC_w*eta
    w->w+v
    b->b-eta*gradC_b
    """
    def __init__(self,learning_rate,momentum=0,reg=reg.Regularization()):
        self.eta = learning_rate
        self.mu = momentum
        self.velocity = None
        self.reg = reg

    def update(self,w,b):

        weights,grad_weights = w
        biases, grad_biases  = b
        eta = self.eta
        #update biases as in SGD
        b = [ b-eta*db for b,db in zip(biases, grad_biases)]

        # init velocity to all zeros
        if not self.velocity:
            self.velocity = [ np.zeros(w.shape) for w in weights]
        else:
            #scale velocity by friction
            self.velocity = [v*self.mu for v in self.velocity]


        #perform update on velocity (using regularization if passed in)
        self.velocity = [ v-eta*dw-(eta*self.reg.derivative(w)) for v,w,dw in zip(self.velocity,weights,grad_weights)]
        #update weights and return
        return ([w+dv for w,dv in zip(weights,self.velocity)],b)
