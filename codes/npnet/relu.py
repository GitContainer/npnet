import numpy as np


class ReLU():
    def __init__(self, shape):
        '''初始化参数'''
        self.eta = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape

    def forward(self, x):
        '''前向传播'''
        self.x = x
        return np.maximum(x, 0)

    def backward(self, eta):
        '''求loss对输入的梯度'''
        self.eta = eta
        # dL/dx = dL/dy * dy/dx = eta * (1 if >0 else 0)
        self.eta[self.x < 0] = 0
        return self.eta
