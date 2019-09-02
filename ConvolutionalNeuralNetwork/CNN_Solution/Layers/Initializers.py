import numpy as np

class Constant:
    def __init__(self, const_value):
        self.const_value = const_value

    def initialize (self,weights_shape, fan_in,fan_out):
        return self.const_value*np.ones (weights_shape)

class UniformRandom ():
    def __init__(self):
        self.initializer = None
    def initialize (self, weights_shape, fan_in,fan_out):
        self.initializer = np.random.rand (fan_in, fan_out)
        return  self.initializer

class Xavier ():
    def __init__(self):
        self.initializer = None

    def initialize (self, weights_shape, fan_in, fan_out):

        sum = fan_in  + fan_out
        variance = np.sqrt(2/sum)
        self.initializer = np.random.normal (0, variance, weights_shape)
        return self.initializer


class He ():
    def __init__(self):
        self.initializer = None

    def initialize (self, weights_shape, fan_in,fan_out):
        variance = np.sqrt(2/fan_in)
        self.initializer = np.random.normal(0, variance, weights_shape)
        return  self.initializer

