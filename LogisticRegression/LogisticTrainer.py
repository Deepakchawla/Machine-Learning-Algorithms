import numpy as np
import h5py
import pandas as pd


class LogisticTrainer:

    def __init__(self):

        # Learning Rate
        self.l_rate = 0.0001

        # Total iterations
        self.iterations = 300000

    def trains(self, x_data_train, y_data_train, theta_vector):

        x_data_train = np.column_stack((np.ones((x_data_train.shape[0], 1), dtype=float), x_data_train))
        for i in range(self.iterations):
            temp = ((np.dot(x_data_train, theta_vector)) - y_data_train)
            temp = np.dot(np.transpose(x_data_train), temp)
            temp = np.dot(temp, self.l_rate) / len(x_data_train)
            theta_vector = theta_vector - temp
        return theta_vector

    def classify(self, x_data_test, theta_vector):

        y_prediction = np.zeros((x_data_test.shape[0], 1), dtype=float)
        x_data_test = np.column_stack((np.ones((x_data_test.shape[0], 1)), x_data_test))
        z = np.dot(x_data_test, theta_vector)
        sigmoid = np.array(1 / (1 + np.exp(-z)))
        for i in (range(0, len(sigmoid))):
            if round(sigmoid[i][0], 2) <= 0.05:
                y_prediction[i][0] = 0
            else:
                y_prediction[i][0] = 1

        return y_prediction

    def accuracy(self, y_data, y_pred):

        return 100 - np.mean(np.abs(y_pred - y_data)) * 100

    def writetocsv(self, actual_data, pred_data, file_name):

        data = np.hstack((actual_data, pred_data))
        output_name = pd.DataFrame(data, columns=['actual_data', 'pred_value'])
        output_name.to_csv(file_name + ".csv", sep=',')


def main():
    # df = pd.read_csv('iris.csv')
    # features = ['sepal_length', 'sepal_width', 'petal_width', 'petal_length']
    # predicted_feature = ['species']
    #
    # x_data_set = np.array(pd.DataFrame(df, columns=features))
    # y_data_set = np.array(pd.DataFrame(df, columns=predicted_feature))
    # x_data_set = (x_data_set - x_data_set.mean()) / x_data_set.std()
    #
    # temp = np.zeros((y_data_set.shape[0], 1), dtype=float)
    #
    # for i in range(0, y_data_set.size):
    #     if y_data_set[i][0] == 'setosa':
    #         temp[i][0] = 1
    #     elif y_data_set[i][0] == 'versicolor' or y_data_set[i] == 'virginica':
    #         temp[i][0] = 0
    # y_data_set = temp
    # x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
    #     x_data_set, y_data_set, test_size=0.20, shuffle=False)
    #
    # theta_vector = np.zeros(((x_data_train.shape[1]+1), 1), dtype='f')
    #
    # l_t = LogisticTrainer()
    # parameters = l_t.trains(x_data_train, y_data_train, theta_vector)
    # y_prediction_test = l_t.classify(x_data_test, parameters)
    # y_prediction_train = l_t.classify(x_data_train, parameters)
    # data = np.hstack((y_data_test, y_prediction_test))
    # output_name = pd.DataFrame(data, columns=['actual_data', 'pred_value'])
    # output_name.to_csv("iris_1.csv", sep=',')
    # for i in range(0, y_data_set.size):
    #     if y_data_set[i][0] == 'versicolor':
    #         temp[i][0] = 1
    #     elif y_data_set[i][0] == 'setosa' or y_data_set[i] == 'virginica':
    #         temp[i][0] = 0
    # y_data_set = temp
    #
    # x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
    #     x_data_set, y_data_set, test_size=0.20, shuffle=False)
    #
    # x_data_train = x_data_train / 255.
    # x_data_test = x_data_test / 255.
    # theta_vector = np.zeros(((x_data_train.shape[1]+1), 1), dtype='f')
    #
    # l_t = LogisticTrainer()
    # parameters = l_t.trains(x_data_train, y_data_train, theta_vector)
    # y_prediction_test = l_t.classify(x_data_test, parameters)
    # y_prediction_train = l_t.classify(x_data_train, parameters)
    # data = np.hstack((y_data_test, y_prediction_test))
    # output_name = pd.DataFrame(data, columns=['actual_data', 'pred_value'])
    # output_name.to_csv("iris_2.csv", sep=',')
    #
    # for i in range(0, y_data_set.size):
    #     if y_data_set[i][0] == 'virginica':
    #         temp[i][0] = 1
    #     elif y_data_set[i][0] == 'versicolor' or y_data_set[i] == 'setosa':
    #         temp[i][0] = 0
    #
    # y_data_set = temp
    #
    # x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
    #     x_data_set, y_data_set, test_size=0.20, shuffle=False)
    #
    # x_data_train = x_data_train / 255.
    # x_data_test = x_data_test / 255.
    # theta_vector = np.zeros(((x_data_train.shape[1]+1), 1), dtype='f')

    trainfile = 'train_catvnoncat.h5'
    testfile = 'test_catvnoncat.h5'

    train_dataset = h5py.File(trainfile, "r")
    test_dataset = h5py.File(testfile, "r")

    x_data_train = np.reshape(np.array(train_dataset['train_set_x'][:]), (np.array(train_dataset['train_set_x'][:]).shape[0], -1))
    y_data_train = np.transpose(np.array([train_dataset['train_set_y'][:]]))

    x_data_test = np.reshape(np.array(test_dataset['test_set_x'][:]), (np.array(test_dataset['test_set_x'][:]).shape[0], -1))
    y_data_test = np.transpose(np.array([test_dataset['test_set_y'][:]]))

    x_data_train = x_data_train / 255.
    x_data_test = x_data_test / 255.

    theta_vector = np.zeros(((x_data_train.shape[1] + 1), 1), dtype='f')

    l_t = LogisticTrainer()
    parameters = l_t.trains(x_data_train, y_data_train, theta_vector)
    y_prediction_test = l_t.classify(x_data_test, parameters)
    y_prediction_train = l_t.classify(x_data_train, parameters)

    print("Train accuracy:", l_t.accuracy(y_data_train, y_prediction_train), "%")
    print("Test accuracy:", l_t.accuracy(y_data_test, y_prediction_test), "%")


# data = np.hstack((y_data_test, y_prediction_test))
    # output_name = pd.DataFrame(data, columns=['actual_data', 'pred_value'])
    # output_name.to_csv("iris_3.csv", sep=',')


if __name__ == '__main__':
    main()
