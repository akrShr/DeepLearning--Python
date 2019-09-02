import copy
class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer

    def append_trainable_layer (self, layer):
        self.layers.append(layer)
        layer.initialize (self.weights_initializer, self.bias_initializer)

        layer.set_optimizer(self.optimizer)

    def forward(self):
        input_tensor,self.label_tensor = self.data_layer.forward()# iris data class
        for layer in self.layers: # bc we are passing the data in all the layers
            input_tensor = layer.forward(input_tensor) # passing input using fwd functn
        self.loss_out = self.loss_layer.forward(input_tensor,self.label_tensor) #calculating loss
        self.loss.append(self.loss_out) # SoftMax class
        return self.loss_layer.forward(input_tensor,self.label_tensor) # calculating loss after forward


    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor) # loss layer se error milega and then passing to the backward
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self,iterations):
        for i in range(iterations):
            self.forward ()
            self.backward()

    def test(self,input_tensor): # only for forward
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        prediction = self.loss_layer.predict(input_tensor)
        return prediction



