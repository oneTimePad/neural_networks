import numpy as np

class Regularization(object):
    def __init__(self,lmbda=0,num_samples=1):
        self.lmbda = lmbda
        self.num_samples = num_samples
    def derivative(self,weights):
        return 0


class L2Reg(Regularization):
    def derivative(self,weights):
        w = np.copy(weights)
        return self.lmbda*w/self.num_samples
