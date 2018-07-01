import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import cross_validation


def load_data(file_path):
    # 打开训练用的csv
    with open(file_path) as f:
        training_data_list = f.readlines()
    X = []
    Y = []
    for record in training_data_list:
        # 输入
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        X.append(inputs)
        Y.append(all_values[0])

    return cross_validation.train_test_split(X, Y, test_size=0.25, random_state=0, stratify=Y)


# 测试：
def test_MLPClassifier(*data):
    X_train, X_test, Y_train, Y_test = data
    nn = MLPClassifier(activation='logistic', max_iter=1000, hidden_layer_sizes=(100,))
    nn.fit(X_train, Y_train)

    print('【训练集成绩】Training Score:{0}'.format(nn.score(X_train, Y_train)))
    print('【测试集成绩】Testing  Score:{0}'.format(nn.score(X_test, Y_test)))


if __name__ == '__main__':

    # 开始测试：
    X_train, X_test, Y_train, Y_test = load_data('./mnist_dataset/mnist_train_100.csv')
    test_MLPClassifier(X_train, X_test, Y_train, Y_test)
