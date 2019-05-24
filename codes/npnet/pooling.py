import numpy as np


class MaxPooling():
    def __init__(self, shape, ksize=2, stride=2):
        '''初始化参数'''
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[-1]
        self.index = np.zeros(shape)  # 最大值的索引
        self.output_shape = [
            shape[0],
            int(shape[1] / self.stride),
            int(shape[2] / self.stride), self.output_channels
        ]

    def forward(self, x):
        '''前向传播'''
        out = np.zeros([
            x.shape[0],
            int(x.shape[1] / self.stride),
            int(x.shape[2] / self.stride), self.output_channels
        ])

        # batch_size
        for b in range(x.shape[0]):
            # c_out
            for c in range(self.output_channels):
                # w
                for i in range(0, x.shape[1], self.stride):
                    # h
                    for j in range(0, x.shape[2], self.stride):
                        # 取最大值
                        out[b,
                            int(i / self.stride),
                            int(j / self.stride), c] = np.max(
                                x[b, (i):(i +
                                          self.ksize), (j):(j +
                                                            self.ksize), c])
                        # 记录最大值所在的索引（反向传播时要用）
                        index = np.argmax(
                            x[b, (i):(i + self.ksize), (j):(j +
                                                            self.ksize), c])
                        self.index[b, i + int(index / self.stride), j +
                                   (index %
                                    self.stride), c] = 1  # 最大值处是1，其他地方是0
        return out

    def backward(self, eta):
        '''求loss对输入的梯度'''
        # *是按位相乘
        # self.index相当于起到掩模的作用
        return np.repeat(np.repeat(eta, self.stride, axis=1),
                         self.stride,
                         axis=2) * self.index
