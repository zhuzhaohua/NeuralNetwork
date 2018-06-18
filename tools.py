from myNeuralNetwork import output_nodes
import numpy
import scipy.misc


# 根据训练文件循环训练的方法
def train(n, file_path, epochs):
    # 打开训练用的csv
    training_data_file = open(file_path, 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # 训练神经网络
    index = 0
    for e in range(epochs):
        # 每一次循环训练所有数据
        for record in training_data_list:
            # 输入
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # 输出断言
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            # 训练
            n.train(inputs, targets)

            # ## 分别左右旋转10度然后再训练
            # inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1,
            #                                                       reshape=False)
            # n.train(inputs_plusx_img.reshape(784), targets)
            # inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1,
            #                                                        reshape=False)
            # n.train(inputs_minusx_img.reshape(784), targets)

            index += 1
            print('train : ' + str(index))


# 根据测试文件测试并打分
def test(n, file_path):
    # 打开考试用的csv
    test_data_file = open(file_path, 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # 测验神经网络

    scorecard = []

    # 对所有数据进行测试
    for record in test_data_list:

        all_values = record.split(',')
        # 输出断言
        correct_label = int(all_values[0])
        # 输入数据
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 测试
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        # 输出结果是一个1*10的矩阵 取其中组大值的索引
        print('断言结果：' + str(correct_label) + '实际输出：' + str(label))
        # 如果与断言一致
        if label == correct_label:
            # 成绩单里加个1
            scorecard.append(1)
        else:
            # 成绩单里加个0
            scorecard.append(0)
            pass

        pass

    # 计算考试结果
    scorecard_array = numpy.asarray(scorecard)

    ret = scorecard_array.sum() / scorecard_array.size * 100

    return ret


def load_png(file_path):
    print("loading ... ", file_path)
    img_array = scipy.misc.imread(file_path, flatten=True)

    # 灰度处理
    img_data = 255.0 - img_array.reshape(784)

    # 数据标准化（0.01 to 1.0）
    img_data = (img_data / 255.0 * 0.99) + 0.01
    # print(numpy.min(img_data))
    # print(numpy.max(img_data))

    return img_data
