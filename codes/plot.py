import matplotlib.pyplot as plt


def read_log(plot_type='batch', path='plot.log'):
    '''读取并解析log文件'''
    result = []
    lines = open(path, 'r').readlines()
    for line in lines:
        part = [str.strip(l) for l in line.split(',')]
        if part[0][1] == plot_type[0]:
            if plot_type == 'batch':
                epoch = int(str.strip(part[1].split('=')[1]))
                batch = int(str.strip(part[2].split('=')[1]))
                avg_batch_acc = float(str.strip(part[3].split('=')[1]))
                avg_batch_loss = float(str.strip(part[4].split('=')[1]))
                learning_rate = float(str.strip(part[5].split('=')[1]))
                result.append([
                    epoch, batch, avg_batch_acc, avg_batch_loss, learning_rate
                ])
            elif plot_type == 'epoch':
                epoch = int(str.strip(part[1].split('=')[1]))
                epoch_acc = float(str.strip(part[2].split('=')[1]))
                avg_epoch_loss = float(str.strip(part[3].split('=')[1]))
                result.append([epoch, epoch_acc, avg_epoch_loss])
            elif plot_type == 'validation':
                epoch = int(str.strip(part[1].split('=')[1]))
                val_acc = float(str.strip(part[2].split('=')[1]))
                avg_val_loss = float(str.strip(part[3].split('=')[1]))
                result.append([epoch, val_acc, avg_val_loss])
    return result


def plot_info(info, info_type='batch'):
    '''根据解析出来的信息画图'''
    if info_type == 'batch':
        x = [(i[0] * 468 + i[1]) for i in info]
        acc = [i[2] for i in info]
        loss = [i[3] for i in info]
        lr = [i[4] for i in info]
        plt.figure(), plt.plot(x, acc), plt.grid()
        plt.xlabel('batch_iter'), plt.ylabel('acc'), plt.title('batch_acc')
        plt.savefig('batch_acc.jpg', dpi=100)

        plt.figure(), plt.plot(x, loss), plt.grid()
        plt.xlabel('batch_iter'), plt.ylabel('loss'), plt.title('batch_loss')
        plt.savefig('batch_loss.jpg', dpi=100)

        plt.figure(), plt.plot(x, lr), plt.grid()
        plt.xlabel('batch_iter'), plt.ylabel('lr'), plt.title('batch_lr')
        plt.savefig('batch_lr.jpg', dpi=100)
        plt.show()
    elif info_type == 'epoch':
        x = [i[0] for i in info]
        acc = [i[1] for i in info]
        loss = [i[2] for i in info]
        plt.figure(), plt.stem(x, acc), plt.grid()
        plt.xlabel('epoch_iter'), plt.ylabel('acc'), plt.title('epoch_acc')
        plt.savefig('epoch_acc.jpg', dpi=100)

        plt.figure(), plt.stem(x, loss), plt.grid()
        plt.xlabel('epoch_iter'), plt.ylabel('loss'), plt.title('epoch_loss')
        plt.savefig('epoch_loss.jpg', dpi=100)
        plt.show()
    elif info_type == 'validation':
        x = [i[0] for i in info]
        acc = [i[1] for i in info]
        loss = [i[2] for i in info]
        plt.figure(), plt.stem(x, acc), plt.grid()
        plt.xlabel('val_iter'), plt.ylabel('acc'), plt.title('val_acc')
        plt.savefig('val_acc.jpg', dpi=100)

        plt.figure(), plt.stem(x, loss), plt.grid()
        plt.xlabel('val_iter'), plt.ylabel('loss'), plt.title('val_loss')
        plt.savefig('val_loss.jpg', dpi=100)
        plt.show()


if __name__ == '__main__':
    batch_info = read_log(plot_type='batch', path='plot.log')
    epoch_info = read_log(plot_type='epoch', path='plot.log')
    val_info = read_log(plot_type='validation', path='plot.log')
    plot_info(batch_info, info_type='batch')
    plot_info(epoch_info, info_type='epoch')
    plot_info(val_info, info_type='validation')
