import numpy as np
from functools import reduce
import math


class FullyConnect():
    def __init__(self, shape, output_num=2):
        '''初始化FC层'''
        self.input_shape = shape
        self.batchsize = shape[0]

        # 相当于Flatten操作，input_len是第二维及其以后的乘积（参数量）
        input_len = int(reduce(lambda x, y: x * y, shape[1:]))

        # 参数初始化
        self.weights = np.random.standard_normal(
            (input_len, output_num)) * math.sqrt(2 / input_len)
        self.bias = np.zeros(output_num)

        self.output_shape = [self.batchsize, output_num]
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self, x):
        '''正向传播'''
        self.x = x.reshape([self.batchsize, -1])  # Flatten
        # 线性加权传播（self.bias会自动扩充维度）
        output = np.dot(self.x, self.weights) + self.bias
        return output

    def backward(self, eta):
        '''求loss对w、b和x的梯度（返回值是loss对x的梯度）'''
        # 计算出对参数的梯度，以在backward()中更新参数
        # 对每个样本
        for i in range(eta.shape[0]):
            col_x = self.x[i][:, np.newaxis]  # 输入值（列向量）
            eta_i = eta[i][:, np.newaxis].T  # dL/dy（转置后得到行向量）
            # 梯度累加
            # dl/dw = dy/dw * dL/dy = x * eta
            self.w_gradient += np.dot(col_x, eta_i)
            # dL/db = dL/dy * dy/db = eta * 1 = eta
            self.b_gradient += eta_i.reshape(self.b_gradient.shape)

        # 计算出loss对输入的导数
        # dL/dx = dL/dy * dy/dx = eta * w
        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.input_shape)

        return next_eta

    def update(self, alpha=0.00001, weight_decay=0.0004):
        '''更新参数'''
        # weight_decay：L2正则化
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        # 梯度下降
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias
        # 清零
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
