import numpy as np
import math


class Conv2D():
    def __init__(self, shape, output_channels, ksize=3, stride=1):
        '''初始化参数'''
        self.input_shape = shape  # 输入尺寸 (batch_size, w, h, c)
        self.output_channels = output_channels  # 等于卷积核的个数
        self.input_channels = shape[-1]  # 输入的通道数
        self.batchsize = shape[0]  # batch大小
        self.ksize = ksize  # 核大小
        self.stride = stride  # 步长

        # 初始化weights和bias
        # 权重使用 Kaiming He 提出的初始化方法（高斯分布）
        self.weights = np.random.standard_normal(
            (ksize, ksize, self.input_channels,
             self.output_channels)) * math.sqrt(
                 2 / (ksize * ksize * self.input_channels))
        self.bias = np.zeros(self.output_channels)  # 偏置初始化为零

        if (shape[1] - self.ksize) % self.stride != 0:
            raise Exception("input tensor width can't fit stride")
        if (shape[2] - self.ksize) % self.stride != 0:
            raise Exception("input tensor height can't fit stride")

        # valid方式进行卷积
        self.eta = np.zeros(
            (self.batchsize, int((shape[1] - ksize) / self.stride) + 1,
             int((shape[2] - ksize) / self.stride) + 1, self.output_channels))

        # 梯度
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

        # 输出尺寸 (batch_size, out_w, out_h, c_out)
        self.output_shape = self.eta.shape

    def forward(self, x):
        '''前向传播'''
        col_weights = self.weights.reshape([-1, self.output_channels
                                            ])  # (ksize * ksize * c_in, c_out)

        self.col_image = []
        conv_out = np.zeros(self.output_shape)
        # 对每个样本
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]  # 取出第i个图像 (1, w, h, c)
            self.col_image_i = im2col(
                img_i, self.ksize,
                self.stride)  # # (out_w*out_h ,ksize * ksize * c_in)
            conv_out[i] = np.reshape(
                np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].
                shape)  # 计算卷积 (out_w*out_h, c_out) -> (out_w, out_h, c_out)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def backward(self, eta):
        '''求loss对conv层参数的导数，以及对输入的导数'''
        self.eta = eta
        col_eta = eta.reshape([self.batchsize, -1, self.output_channels])

        # 计算出loss对参数的梯度
        for i in range(self.batchsize):
            # dL/dw = dy/dw * dL/dy = x * eta
            self.w_gradient += np.dot(self.col_image[i].T,
                                      col_eta[i]).reshape(self.weights.shape)
        # dL/db = dL/dy * dy/db = eta * 1 = eta
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        # dL/dx = dL/dy * dy/dx = eta * w
        # 反向的转置卷积，计算出next_eta
        # 先将eta填充，再将kernel转置，最后进行反向的转置卷积
        padding_value = ((self.stride**2 - 1) * self.eta.shape[1] +
                         self.ksize + self.stride *
                         (self.ksize - self.stride - 1)) / 2
        padding_value = math.ceil(padding_value)  # 计算转置卷积的填充值
        pad_eta = np.pad(self.eta, ((0, 0), (padding_value, padding_value),
                                    (padding_value, padding_value), (0, 0)),
                         'constant',
                         constant_values=0)  # 零填充
        flip_weights = np.flipud(np.fliplr(self.weights))  # 转置卷积核
        flip_weights = flip_weights.swapaxes(2, 3)  # 输入输出通道调换
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_eta = np.array([
            im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride)
            for i in range(self.batchsize)
        ])
        # 逆向计算出next_eta
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def update(self, alpha=0.00001, weight_decay=0.0004):
        '''更新参数'''
        # weight_decay：L2正则化（权重衰减系数）
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient  # 梯度下降
        self.bias -= alpha * self.bias  # 梯度下降
        # 梯度归零
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)


def im2col(image, ksize, stride):
    '''输入tensor（如图像）转化为适用于卷积的形式（矩阵）'''
    # image尺寸：(batch_size==1, width ,height, channel)
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, (i):(i + ksize), (j):(j + ksize), :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)  # 转为ndarray格式（集成在第一个维度上）

    return image_col  # (out_w*out_h, ksize*ksize*channel)
