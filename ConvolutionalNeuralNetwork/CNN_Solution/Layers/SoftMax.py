import numpy as np
class SoftMax:
    def __init__(self):
        pass

    def forward (self, input_tensor,label_tensor):
        #x_exp= np.exp(input_tensor- np.max(input_tensor))
        #x_sum= np.sum(x_exp)
        #y_hat= x_exp/x_sum
        self.input_tensor= input_tensor
        self.label_tensor= label_tensor
        self.y_hat= SoftMax.predict(self,self.input_tensor)
        loss_y_hat= np.log(self.y_hat)
        np.place(loss_y_hat,label_tensor!=1, 0.0)
        self.loss= -np.sum(loss_y_hat)
        #loss= np.where(label_tensor==1,(-np.log (np.dot(y_hat,label_tensor)
        return self.loss  ## multiply label tensor with prediction value (y_hat)


    def predict (self,input_tensor):

        x_exp = np.exp(input_tensor - np.max(input_tensor))
        #x_sum= np.sum(x_exp)
        self.y_hat = np.divide(x_exp, np.expand_dims(np.sum(x_exp, axis=1),1)) # why np.dims?
        return self.y_hat

    def backward (self, label_tensor):
        error_tensor= self.y_hat - label_tensor
        return  error_tensor






