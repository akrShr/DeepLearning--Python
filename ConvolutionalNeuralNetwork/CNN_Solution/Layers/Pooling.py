import numpy as np




import numpy as np

class Pooling:
    def __init__(self, stride_shape,pooling_shape ):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self,input_tensor):
        self.x = input_tensor
        batch, channel, y_dim, x_dim= input_tensor.shape
        input_dim,k = self.pooling_shape
        stride_y,stride_x= self.stride_shape
        y_out = np.int((y_dim - input_dim) /stride_y + 1)
        x_out = np.int((x_dim - k) / stride_x + 1)
        out = np.zeros((batch, channel, y_out, x_out))

        for i in range(y_out):
            for j in range(x_out):
                mask = input_tensor[:, :, i * stride_y: i * stride_y + input_dim, j * stride_x: j * stride_x + k]
                out[:, :, i, j] = np.max(mask, axis=(2, 3))
        return out

    def backward(self, error_tensor):
        batch, channel, y_dim, x_dim = self.x.shape
        input_dim,k = self.pooling_shape
        stride_y,stride_x = self.stride_shape
        y_out = np.int((y_dim - input_dim) /stride_y + 1)
        x_out = np.int((x_dim - k) / stride_x + 1)
        error = np.zeros_like(self.x)
        m = 0
        error_tensor = error_tensor.reshape(np.prod(batch*channel*y_out*x_out))

        for a in range(batch):
            for b in range(channel):
                for i in range(y_out):
                    for j in range(x_out):
                        mask = self.x[a, b, i * stride_y:i * stride_y + input_dim, j * stride_x:j * stride_x + k]
                        index = np.argmax(mask)
                        error_mask = error[a, b, i * stride_y:i * stride_y + input_dim, j * stride_x:j * stride_x + k]
                        error_mask[index // k, index % k] += error_tensor[m]
                        m += 1
        return error

