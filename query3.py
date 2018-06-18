'''
使用训练集csv对神经网络进行训练
使用测试集csv对神经网络进行测试
并评估得分

并测试其是否有举一反三的能力
'''
from myNeuralNetwork import n
from myNeuralNetwork import output_nodes
import numpy
import matplotlib.pyplot
import tools

if __name__ == '__main__':
    # 训练
    tools.train(n, './mnist_dataset/mnist_train.csv', 1)

    # 测试
    ret = tools.test(n, './mnist_dataset/mnist_test.csv')

    print("==" * 20)
    print("成绩 = ", ret)

    # 做一个输出，比如[0.99，0.01，，0.01，，0.01...]表示 0
    label = 0
    targets = numpy.zeros(output_nodes) + 0.01
    targets[label] = 0.99
    print(targets)

    # 输入神经网络，反推
    image_data = n.back_query(targets)

    # 加载图像
    matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')

    # 显示图形
    matplotlib.pyplot.show()
