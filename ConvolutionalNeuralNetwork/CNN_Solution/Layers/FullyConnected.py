import numpy as np
from copy import *
class FullyConnected():
#
     def __init__(self,input_size, output_size):
         self.input_size = input_size
         self.output_size = output_size
         self.weights = np.random.uniform(size=(self.input_size+1, self.output_size)) # defining the parameters uniformly
#         #self.bias = np.zeros(output_size)
#         self.input_tensor= []
#         self.output_tensor= []
         self.delta=0.1
#         #self.bias = np.ones ([output_size,1])
#         self.Hint = None
#         self._gradient_weights = np.zeros(self.weights.shape)
         self.optimizer = None
#
     def initialize (self, weights_initializer, bias_initializer):
         weights_without_bias = weights_initializer.initialize((self.input_size, self.output_size),self.input_size,self.output_size)
         self.bias = bias_initializer.initialize((1,self.output_size),1,self.output_size)
         self.weights = np.concatenate((weights_without_bias, self.bias), axis=0)

#         self.weights = weights_initializer.initialize((self.output_size, self.input_size), self.input_size,
#                                                       self.output_size)
#         self.bias = bias_initializer.initialize((self.output_size, 1), 1, self.output_size)
#
#         self.weights = np.hstack ((weights, bias))

#
#
     def set_optimizer (self, optimizer):
         self.optimizer = deepcopy(optimizer)
#
#
#
     def forward(self, input_tensor):
#
         add_one = np.ones((input_tensor.shape[0],1))# adding 1 in the end in the input vector
         mod_input_tensor = np.column_stack((input_tensor,add_one)) # 1D array to 2D array
         self.input_tensor = mod_input_tensor
         self.output_tensor = np.dot(self.input_tensor,self.weights) # w*input+bias
#         #self.data.append(input_tensor)
#         #return output_tensor
#         bias_input = np.ones ((input_tensor.shape[0],1))
#         self.Hint = np.append ((input_tensor,bias_input))
#         self.linear_transform = np.dot (self.Hint, self.weights)
#
         return self.output_tensor

     # def update_parameter (self, error_tensor):
     #    self._gradient_weights = np.dot (error_tensor, np.transpose(self.Hint))
     #    if self.optimizer is not None:
     #        self.weights = self.optimizer.calculate_update (self.delta, self.weights,self._gradient_weights)

     def backward(self, error_tensor):
        self.error_tensor = error_tensor
        error_input = error_tensor
        error_tensor_next_layer = np.dot(error_input,np.transpose(self.weights)) # grad wrt input
        error_tensor_del = np.delete(error_tensor_next_layer,-1, axis=1) # delete the row
        self.gradient= self.get_gradient_weights()
        #self.weights = self.weights - self.delta* gradient

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights,self.gradient)

        #self.update_parameter(error_tensor)

        return error_tensor_del

     def get_gradient_weights(self):
        gradient = np.zeros_like(self.weights) # returns zeroes of same shape and type as of weights
        gradient = np.dot(np.transpose(self.input_tensor), self.error_tensor) # gradient wrt to weights (error*input.T)
        return gradient







