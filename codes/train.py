import time
import struct
from glob import glob
import numpy as np
from model import Model_5LayersCNN


def standardize(seq):
    '''标准化'''
    centerized = seq - np.mean(seq)
    normalized = centerized / np.std(centerized)
    return normalized


def load_mnist(path, kind='train'):
    """
    读取MNIST数据集
    path：数据集路径
    kind：数据集类型，训练集'train'或测试集't10k'
    """
    images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    # 标准化
    images = images.astype(np.float64)
    for i in range(images.shape[0]):
        images[i] = standardize(images[i])

    # 行样本
    # 返回的训练集尺寸 (60000, 784) ，标签尺寸(60000,)
    # 返回的测试集尺寸(10000, 784) ，标签尺寸(10000,)
    return images, labels


def init_logs(path='plot.log'):
    '''初始化日志文件'''
    with open(path, 'w') as f:
        f.write('')


def write_logs(str, path='plot.log'):
    '''写入到日志文件'''
    with open(path, 'a') as f:
        f.write(str + '\n')


if __name__ == '__main__':
    # 初始化日志文件
    init_logs(path='plot.log')

    # 读取训练数据和测试数据
    images, labels = load_mnist(path='./datasets/mnist', kind='train')
    test_images, test_labels = load_mnist(path='./datasets/mnist', kind='t10k')

    # 批大小
    batch_size = 128  # 训练时的batch_size，一个batch有128个样本
    epoch_num = 3  # 遍历数据集3次
    learning_rate = 1e-4  # 学习率
    weight_decay = 5e-4  # 权重衰减

    # 模型定义
    model = Model_5LayersCNN(batch_size)

    # epoch
    for epoch in range(epoch_num):
        # 减缓学习率
        if epoch == 1:
            learning_rate *= 0.2
        elif epoch == 2:
            learning_rate *= 0.2

        # epoch
        epoch_loss = 0
        epoch_acc = 0

        # batch
        for i in range(int(images.shape[0] / batch_size)):
            # 指标
            batch_loss = 0
            batch_acc = 0

            img = images[(i * batch_size):((i + 1) * batch_size)].reshape(
                [batch_size, 28, 28,
                 1])  # [batch_size, 784] -> [batch_size, 28, 28, 1]
            label = labels[(i * batch_size):((i + 1) *
                                             batch_size)]  # [batch_size, ]

            # 正向传播
            model.forward(img)
            # 计算损失
            loss = model.cal_loss(label)
            batch_loss += loss
            epoch_loss += loss

            # 统计出分类正确的个数
            num = model.get_correct_num(label)
            batch_acc += num
            epoch_acc += num

            # 计算梯度
            model.backward()

            # 根据梯度来更新权重
            model.update(lr=learning_rate, decay=weight_decay)

            # 打印batch信息
            log_str = '[batch]' + time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime()
            ) + ",  epoch= %d,  batch= %d,  avg_batch_acc= %.4f,  avg_batch_loss= %.4f,  learning_rate= %f" % (
                epoch, i, batch_acc / batch_size, batch_loss / batch_size,
                learning_rate)
            print(log_str)
            write_logs(log_str, path='plot.log')

        # 打印epoch信息
        log_str = '[epoch] ' + time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        ) + ",  epoch= %d ,  epoch_acc= %.4f,  avg_epoch_loss= %.4f" % (
            epoch, epoch_acc / images.shape[0], epoch_loss / images.shape[0])
        print(log_str)
        write_logs(log_str, path='plot.log')

        # 用测试数据集来验证精度
        val_loss = 0
        val_acc = 0
        for i in range(int(test_images.shape[0] / batch_size)):
            img = test_images[(i * batch_size):((i + 1) * batch_size)].reshape(
                [batch_size, 28, 28,
                 1])  # [batch_size, 784] -> [batch_size, 28, 28, 1]
            label = test_labels[(i *
                                 batch_size):((i + 1) *
                                              batch_size)]  # [batch_size, ]
            # 正向传播
            model.forward(img)
            # 计算损失
            loss = model.cal_loss(label)
            val_loss += loss

            # 统计出分类正确的个数
            val_acc += model.get_correct_num(label)

        log_str = '[validation]' + time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(
            )) + ",  epoch= %d ,  val_acc= %.4f,  avg_val_loss= %.4f" % (
                epoch, val_acc / test_images.shape[0],
                val_loss / test_images.shape[0])
        print(log_str)
        write_logs(log_str, path='plot.log')
