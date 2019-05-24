import numpy as np
from npnet.conv import Conv2D
from npnet.fc import FullyConnect
from npnet.pooling import MaxPooling
from npnet.softmax import Softmax
from npnet.relu import ReLU


class Model_5LayersCNN():
    def __init__(self, batch_size):
        '''初始化参数'''
        self.batch_size = batch_size
        self.conv1 = Conv2D(shape=[batch_size, 28, 28, 1],
                            output_channels=5,
                            ksize=3,
                            stride=1)

        self.relu1 = ReLU(shape=self.conv1.output_shape)
        self.pool1 = MaxPooling(shape=self.relu1.output_shape,
                                ksize=2,
                                stride=2)

        self.conv2 = Conv2D(shape=self.pool1.output_shape,
                            output_channels=10,
                            ksize=3,
                            stride=2)
        self.relu2 = ReLU(shape=self.conv2.output_shape)

        self.conv3 = Conv2D(shape=self.relu2.output_shape,
                            output_channels=10,
                            ksize=1,
                            stride=1)
        self.relu3 = ReLU(shape=self.conv3.output_shape)
        self.pool3 = MaxPooling(shape=self.relu3.output_shape,
                                ksize=2,
                                stride=2)

        self.fc4 = FullyConnect(shape=self.pool3.output_shape, output_num=30)
        self.relu4 = ReLU(shape=self.fc4.output_shape)

        self.fc5 = FullyConnect(shape=self.relu4.output_shape, output_num=10)
        self.sf5 = Softmax(shape=self.fc5.output_shape)

        # 组织顺序
        self.seq = [
            self.conv1, self.relu1, self.pool1, self.conv2, self.relu2,
            self.conv3, self.relu3, self.pool3, self.fc4, self.relu4, self.fc5,
            self.sf5
        ]

    def get_correct_num(self, label):
        '''获得分类正确的个数'''
        num = 0
        for i in range(self.batch_size):
            if np.argmax(self.sf5.softmax[i]) == label[i]:
                num += 1
        return num

    def cal_loss(self, label):
        '''计算损失值'''
        self.loss = self.sf5.cal_loss(label)
        return self.loss

    def forward(self, img):
        '''正向传播'''
        self.out = img
        for layer in self.seq:
            self.out = layer.forward(self.out)
        return self.out

    def backward(self):
        '''反向传播'''
        eta = self.sf5.softmax
        for layer in reversed(self.seq):
            eta = layer.backward(eta)

    def update(self, lr=1e-5, decay=1e-4):
        '''更新参数'''
        for layer in reversed(self.seq):
            if 'update' in dir(layer):
                layer.update(alpha=lr, weight_decay=decay)
