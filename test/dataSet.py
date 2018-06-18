import numpy
import matplotlib.pyplot

# 打开训练用数据的100件csv
data_file = open("../mnist_dataset/mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()


# 查看一下件数
# print(len(data_list))


# 取第一件看一下大概的样子 第一个数字是该明细所表示的数字，后面28*28个像素点的值
# print(data_list[1])


# 将其转换成28*28的矩阵image_array
all_values = data_list[1].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
# 渲染图形
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
# 显示图形
matplotlib.pyplot.show()


# 将各像素值降低为0~1之间的小数
scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(scaled_input)



# 输出的样式 应该是1*10的矩阵，表现每一个数字的可能性
onodes = 10
targets = numpy.zeros(onodes) + 0.01
# 第一个数代表这个矩阵所表示的值，将其赋值给输出所对应的值为0.99
targets[int(all_values[0])] = 0.99

print(targets)