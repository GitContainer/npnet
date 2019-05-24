import numpy as np


class Softmax():
    def __init__(self, shape):
        '''初始化参数'''
        self.softmax = np.zeros(shape)  # 输出值 [batch_size, out_num]
        self.eta = np.zeros(shape)  # 误差
        self.batchsize = shape[0]

    def cal_loss(self, label):
        '''计算softmax loss'''
        self.label = label
        self.loss = 0
        # 对每个样本
        for i in range(self.batchsize):
            # softmax + cross entropy loss = softmax loss
            self.loss += -np.log(self.softmax[i, label[i]])
        return self.loss

    def forward(self, x):
        '''计算softmax输出概率'''
        exp_x = np.zeros(x.shape)
        self.softmax = np.zeros(x.shape)
        for i in range(self.batchsize):
            x[i, :] -= np.max(x[i, :])  # 这一步是为了防止过大，取指数时溢出
            exp_x[i] = np.exp(x[i])
            self.softmax[i] = exp_x[i] / np.sum(exp_x[i])  # 计算出softmax之后的输出值
        return self.softmax

    def backward(self, eta):
        '''计算loss对输入的梯度'''
        self.eta = eta.copy()
        # 对每个样本
        for i in range(self.batchsize):
            # dL/d{x_i} = softmax_out_value_{i} - y_{i}
            self.eta[i, self.label[i]] -= 1
        return self.eta
