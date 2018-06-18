from neuralNetwork import neuralNetwork


# 输入节点数量
input_nodes = 784
# 隐藏节点数量
hidden_nodes = 200
# 输出节点数量
output_nodes = 10

# 学习率
learning_rate = 0.1

# 按照各节点数量以及学习率创建神经网络
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)