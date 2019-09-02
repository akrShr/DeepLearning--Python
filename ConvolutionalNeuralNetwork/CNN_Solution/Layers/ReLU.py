import numpy as np
class ReLU:
    def __init__(self):
        self.data=[]
    def forward (self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum (0, input_tensor) # relu-- max(0,x)
    def backward (self,error_tensor):
        relu_grad = self.input_tensor>0
        return error_tensor*relu_grad # dl/dy. dy/dx (Error. Input_tensor)
