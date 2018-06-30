'''
使用训练集csv对神经网络进行训练
使用测试集csv对神经网络进行测试
并评估得分
'''
from myNeuralNetwork import n
import tools
import shelve

if __name__ == '__main__':
    # 训练
    # 参数 1 神经网络
    # 参数 2 训练用的数据集
    # 参数 3 训练几遍
    tools.train(n, './mnist_dataset/mnist_train.csv', 1)

    # 测试
    # 参数 1 神经网络
    # 参数 2 测试用数据集
    ret = tools.test(n, './mnist_dataset/mnist_test.csv')

    print("==" * 20)
    print("成绩 = ", ret)

