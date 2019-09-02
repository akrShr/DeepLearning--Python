from os import error

import numpy as np
from scipy.signal import correlate2d
from scipy.signal import correlate, convolve
from copy import deepcopy

class Conv:
    def __init__(self,stride_shape,convolution_shape,num_kernels):

        if len(stride_shape)== 1:
            self.conv1D=True
            stride_shape.append(1)
            convolution_shape +=(1,)
        else:
            self.conv1D = False
        self.stride_shape=stride_shape
        self.convolution_shape=convolution_shape #Filter
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(0, 1, (self.num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(0, 1, (self.num_kernels, 1))
        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)
        self.weight_optimizer = None
        self.bias_optimizer = None




    def forward(self,input_tensor):
        if len(input_tensor.shape) == 3: # 1d convolution (no x) only (batch, channel, y)
            input_tensor = np.reshape(input_tensor, (input_tensor.shape + (1,)))

        self.input_tensor = input_tensor # input_tensor is 4d
        batch_size, channels, ydims, xdims = input_tensor.shape

        #Padding for gradient calculations
        kernel_channel, kernel_y, kernel_x = self.convolution_shape
        pady = (kernel_y-1) // 2 # padding form.
        padx = (kernel_x-1) // 2
        self.padded_input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (pady, kernel_y - pady - 1), (padx, kernel_x - padx - 1)), # length and sequence of padding
                              'constant', constant_values=0)

        out_ydims = int(np.ceil(ydims / self.stride_shape[0]))
        if self.conv1D:
            out_xdims = 1
        else:
            out_xdims = int(np.ceil(xdims / self.stride_shape[1]))

        out = np.zeros((batch_size, self.num_kernels, out_ydims, out_xdims))

        self.paded_input_tensor_columnwise = []
        for batch in range(batch_size):   # taking one image from batch
            input_img=input_tensor[batch]
            padded_input_img = self.padded_input_tensor[batch]
            self.paded_input_tensor_columnwise.append(self.image_column_vectorize(padded_input_img,self.stride_shape[0],self.stride_shape[1],kernel_y,kernel_x))
            for kernel in range(self.num_kernels):
                kernel_w=self.weights[kernel]  # weight of 1st filter
                filt_out=np.zeros_like(input_img)  # because of same correlation
                for channel in range(channels):
                    filt_out[channel]=correlate(input_img[channel],kernel_w[channel],mode='same')  # correlation
                filt_out=np.sum(filt_out,axis=0)  # output of 3 channels, taking mean of channel (3,y,x) is the output shape
                strided_filt_out = np.array(filt_out[0::self.stride_shape[0], 0::self.stride_shape[1]])+self.bias[kernel] # striding in y and x dirn
                out[batch][kernel]=strided_filt_out

        self.paded_input_tensor_columnwise = np.array(self.paded_input_tensor_columnwise)

        if self.conv1D: # reshaping 1d because no x_dim
            out=out.reshape(batch_size, self.num_kernels, out_ydims)

        return out

    def image_column_vectorize(self, img, stride_y, stride_x, kernel_y, kernel_x): # RGB channel values
        c, y, x = img.shape
        image_columnwise = []
        if self.conv1D:
            for i in range(0, y - kernel_y + 1, stride_y):
                col = np.reshape(img[:, i:i + kernel_y, :], [-1])
                image_columnwise.append(col)
        else:
            for i in range(0, y - kernel_y + 1, stride_y):
                for j in range(0, x - kernel_x + 1, stride_x):
                    col = np.reshape(img[:, i:i + kernel_y, j:j + kernel_x], [-1])
                    image_columnwise.append(col)

        image_columnwise = np.array(image_columnwise)
        return image_columnwise # returning all channel values in a column

    def backward(self,error_tensor):
        self.error_tensor=error_tensor # error_tensor for next layer

        out=np.zeros_like(self.input_tensor)
        batch_size, channels, ydims, xdims = self.input_tensor.shape
        upsampled_error_tensor=np.zeros((batch_size,self.num_kernels,ydims,xdims))

        if self.conv1D:
            upsampled_error_tensor=upsampled_error_tensor.reshape(batch_size,self.num_kernels,ydims,xdims)
            for i in range(error_tensor.shape[2]):
                upsampled_error_tensor[:, :, i * self.stride_shape[0], 0] = error_tensor[:, :, i]

        else:
            for i in range(error_tensor.shape[2]):
                for j in range(error_tensor.shape[3]):
                    upsampled_error_tensor[:, :, i * self.stride_shape[0], j * self.stride_shape[1]] = error_tensor[:, :, i, j]

        self.bias_gradient = np.sum(upsampled_error_tensor, axis=(0, 2, 3))[:, np.newaxis]


        #backward_weight = self.weights[:, :, ::-1, ::-1]  # interchanging y and x # convolution
        backward_weight = np.swapaxes(self.weights, 0, 1) # because no channel in the weight o/p so swapping kernel and channel

        for batch in range(batch_size):
            error_img=upsampled_error_tensor[batch]
            for channel in range(channels):
                kernel_w = backward_weight[channel]
                filt_out=np.zeros_like(error_img)
                for kernel in range(self.num_kernels):
                    filt_out[kernel]=convolve(error_img[kernel],kernel_w[kernel],mode='same')
                filt_out=np.sum(filt_out,axis=0)
                out[batch][channel]=filt_out  # opposite of fwd, [batch, channel]

        if self.conv1D:
            out=out.reshape(batch_size, channels, ydims)

        """
        Weight and bias gradients calculations here
        """
        col_error = np.reshape(error_tensor, [batch_size, self.num_kernels, -1])
        #self.bias_gradient += np.reshape(np.sum(col_error, axis=(0, 2)), (self.num_kernels, -1))


        for batch in range(batch_size):
            self.weights_gradient += np.dot(col_error[batch], self.paded_input_tensor_columnwise[batch]).reshape(self.weights.shape)

        if self.weight_optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.weights_gradient)
        if self.bias_optimizer:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.bias_gradient)

        self.weights_gradient = np.zeros_like(self.weights)
        for i in range (batch_size):
            for j in range (self.error_tensor.shape[1]):
                x = correlate(self.padded_input_tensor[i],upsampled_error_tensor[i, j][np.newaxis, ...],'valid')
                self.weights_gradient[j] += x
        return out

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * self.convolution_shape[1] * self.convolution_shape[2]
        self.weights = weights_initializer.initialize((self.num_kernels, *self.convolution_shape), fan_in, fan_out)
        self.bias = bias_initializer.initialize((self.num_kernels, 1), 1, self.num_kernels)



    def set_optimizer(self, optimizer):
        self.optimizer = deepcopy(optimizer)
        self.bias_optimizer = deepcopy(optimizer)


    def get_gradient_weights(self):
        return self.weights_gradient


    def get_gradient_bias(self):
        return self.bias_gradient
