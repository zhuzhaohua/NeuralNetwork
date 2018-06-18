"""
使用训练集csv对神经网络进行训练
使用测试位图对神经网络进行测试
并评价是否正确
"""
from myNeuralNetwork import n
import numpy
import matplotlib.pyplot
import tools


if __name__ == '__main__':
    # 训练
    tools.train(n, './mnist_dataset/mnist_train.csv', 1)

    # 待测试的图片
    image_file_name = './my_own_images/2828_my_own_2.png'

    # 断言的值
    label = int(image_file_name[-5:-4])

    # 将图片解析成28*28的灰度数组
    img_data = tools.load_png(image_file_name)

    # 加载图片
    matplotlib.pyplot.imshow(img_data.reshape(28, 28), cmap='Greys', interpolation='None')
    # 显示图片
    matplotlib.pyplot.show()

    # 输入神经网络 查看结果
    outputs = n.query(img_data)
    print(outputs)

    # 输出值最大的那个项目就是结果
    label2 = numpy.argmax(outputs)

    if label == label2:
        print("识别成功！")
    else:
        print("识别失败！")

