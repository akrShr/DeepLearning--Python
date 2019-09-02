import pdb
import numpy as np


class Flatten():

    def __init__(self):
        pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor.shape[0]
        self.input_shape = input_tensor[0].shape

        input_tensor = input_tensor.reshape(self.input_tensor, np.prod(self.input_shape))

        return input_tensor

    def backward(self, error_tensor):
        # pdb.set_trace()
        error_tensor = error_tensor.reshape(self.input_tensor, *self.input_shape)
        return error_tensor
