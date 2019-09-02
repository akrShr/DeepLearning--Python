import numpy as np
import math


class Sgd():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update (self, weight_tensor, gradient_tensor):

        weight_tensor = weight_tensor - self.learning_rate*gradient_tensor
        self.weight_tensor = weight_tensor

        return self.weight_tensor


class SgdWithMomentum():
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self. velocity_lower = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.velocity_lower = self.momentum*self.velocity_lower - (self.learning_rate*gradient_tensor)
        weight_tensor = weight_tensor + self.velocity_lower # update weight
        #self.velocity_lower = velocity
        self.weight_tensor = weight_tensor

        return  self.weight_tensor

class Adam():
    def __init__(self, learning_rate,beta1, beta2):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.velocity_lower = 0
        self.r_lower = 0
        self.iterations = 1



    def calculate_update(self, weight_tensor, gradient_tensor):


        self.velocity_lower = (self.beta1*self.velocity_lower)+ (1-self.beta1)*gradient_tensor
        self.r_lower = (self.beta2*self.r_lower) + (1-self.beta2)* np.dot(gradient_tensor,gradient_tensor)
        velocity_lower = self.velocity_lower / (1-np.power(self.beta1,self.iterations))
        r_lower = self.r_lower / (1-np.power(self.beta2,self.iterations))
        self.iterations += 1
        weight_tensor = weight_tensor - self.learning_rate*(velocity_lower + 10e-8)/ (np.sqrt(r_lower)+ 10e-8)
        self.weight_tensor = weight_tensor

        return  self.weight_tensor

