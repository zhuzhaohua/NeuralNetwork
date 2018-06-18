import numpy
import scipy.special


# 神经网络类
class neuralNetwork:

    # 构造函数
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 输入节点数量
        self.inodes = inputnodes
        # 隐藏节点数量
        self.hnodes = hiddennodes
        # 输出节点数量
        self.onodes = outputnodes

        # 随机权重！
        # 权重矩阵（正态概率分布随机权重矩阵）
        # wih 是输入层到隐藏层的
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        # who 是隐藏层到输出层的
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # 设置学习率
        self.lr = learningrate

        # 设置激活（抑制）函数special函数 平滑的 0到1的曲线
        self.activation_function = lambda x: scipy.special.expit(x)
        # 反向传递的激活（抑制）函数
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
        pass

    # 训练学习的方法
    def train(self, inputs_list, targets_list):
        # 输入矩阵
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 断言矩阵
        targets = numpy.array(targets_list, ndmin=2).T

        # 输入矩阵点乘权重
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 过激活函数
        hidden_outputs = self.activation_function(hidden_inputs)

        # 过激活函数后的输出矩阵就是输出层的输入矩阵 再点乘权重
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 过激活函数 得到最终输出
        final_outputs = self.activation_function(final_inputs)

        # 实际输出与断言的误差
        output_errors = targets - final_outputs
        # 按权重反向传播误差 至隐藏层误差
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 调整隐藏层至输出层权重矩阵
        # 调整值 =  学习率*（输出误差*输出*（1-输出）· 隐藏层输出转置）
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # 调整输入层至隐藏层权重矩阵
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # 识别方法
    def query(self, inputs_list):
        # 输入矩阵
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 输入矩阵点乘权重
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 过激活函数
        hidden_outputs = self.activation_function(hidden_inputs)

        # 过激活函数后的输出矩阵就是输出层的输入矩阵 再点乘权重
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 过激活函数 得到最终输出
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # backquery the neural network
    # we'll use the same termnimology to each item,
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def back_query(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs
