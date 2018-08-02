import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import xlrd
import csv

class LogisticTrainer:

    def __init__(self, l_rate):

        # Learning Rate
        self.l_rate = l_rate

        # Total iterations
        self.iterations = 2000

    def trains(self, x_data_train, y_data_train, theta_vector):
        lamda = 0.0010
        x_data_train = np.column_stack((np.ones((x_data_train.shape[0], 1), dtype=float), x_data_train))
        for i in range(self.iterations):
            z = np.dot(x_data_train, theta_vector)
            sigmoid = (1 / (1 + np.exp(-z)))
            a = sigmoid - y_data_train
            temp = np.dot( x_data_train.T, a)
            temp = np.dot(temp, self.l_rate) / len(x_data_train)
            theta_vector = theta_vector - temp
        return theta_vector

    def classify(self, x_data_test, theta_vector):

        y_prediction = np.zeros((x_data_test.shape[0], 1), dtype=float)
        x_data_test = np.column_stack((np.ones((x_data_test.shape[0], 1)), x_data_test))
        z = np.dot(x_data_test, theta_vector)
        sigmoid = np.array(1 / (1 + np.exp(-z)))
        for i in (range(0, len(sigmoid))):
            if round(sigmoid[i][0], 2) <= 0.5:
                y_prediction[i][0] = 0
            else:
                y_prediction[i][0] = 1
        return y_prediction

    def accuracy(self, y_data_test, y_pred_test):
        total_error = 0
        error = []
        for i in range(0, len(y_data_test)):
            total_error = total_error + abs((y_pred_test[i] - y_data_test[i]) / y_data_test[i])
        total_error = (total_error / len(y_data_test))
        accuracy = 1 - total_error
        return accuracy * 100

    def writetocsv(self, actual_data, pred_data, file_name):

        data = np.hstack((actual_data, pred_data))
        output_name = pd.DataFrame(data, columns=['actual_data', 'pred_value'])
        output_name.to_csv(file_name + ".csv", sep=',')

    def xlsvtocsv(self, xlsx_file, csv_file):

        wb = xlrd.open_workbook(xlsx_file)
        sh = wb.sheet_by_name(wb.sheet_names()[0])
        your_csv_file = open(csv_file, 'w')
        wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

        for rownum in range(sh.nrows):
            wr.writerow(sh.row_values(rownum))

        your_csv_file.close()


def main():

    # df = pd.read_csv('iris.csv')
    # features = ['sepal_length', 'sepal_width', 'petal_width', 'petal_length']
    # predicted_feature = ['species']
    #
    # x_data_set = np.array(pd.DataFrame(df, columns=features))
    # y_data_set = np.array(pd.DataFrame(df, columns=predicted_feature))
    #
    # x_data_set = (x_data_set - x_data_set.mean()) / x_data_set.std()
    #
    # temp = np.zeros((y_data_set.shape[0], 1), dtype=float)
    #
    # for i in range(0, y_data_set.size):
    #     if y_data_set[i][0] == 'setosa':
    #         temp[i][0] = 1
    #     elif y_data_set[i][0] == 'versicolor' or y_data_set[i] == 'virginica':
    #         temp[i][0] = 0
    #
    # y_data_set = temp
    #
    # x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
    #     x_data_set, y_data_set, test_size=0.20)
    #
    # theta_vector = np.zeros(((x_data_train.shape[1]+1), 1), dtype='f')
    #
    # l_t = LogisticTrainer(0.0004)
    # parameters = l_t.trains(x_data_train, y_data_train, theta_vector)
    # y_prediction_test = l_t.classify(x_data_test, parameters)
    # y_prediction_train = l_t.classify(x_data_train, parameters)
    #
    # data = np.hstack((y_data_test, y_prediction_test))
    # output_name = pd.DataFrame(data, columns=['actual_data', 'pred_value'])
    # output_name.to_csv("iris_1.csv", sep=',')
    #
    # print(float(100 - np.mean(np.abs(y_prediction_train - y_data_train)) * 100))
    # print(float(100 - np.mean(np.abs(y_prediction_test - y_data_test)) * 100))
    #
    # for i in range(0, y_data_set.size):
    #     if y_data_set[i][0] == 'versicolor':
    #         temp[i][0] = 1
    #     elif y_data_set[i][0] == 'setosa' or y_data_set[i] == 'virginica':
    #         temp[i][0] = 0
    # y_data_set = temp
    #
    # x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
    #     x_data_set, y_data_set, test_size=0.20)
    #
    # theta_vector = np.zeros(((x_data_train.shape[1]+1), 1), dtype='f')
    #
    # l_t = LogisticTrainer(0.0004)
    # parameters = l_t.trains(x_data_train, y_data_train, theta_vector)
    # y_prediction_test = l_t.classify(x_data_test, parameters)
    # y_prediction_train = l_t.classify(x_data_train, parameters)
    #
    # data = np.hstack((y_data_test, y_prediction_test))
    #
    # output_name = pd.DataFrame(data, columns=['actual_data', 'pred_value'])
    # output_name.to_csv("iris_2.csv", sep=',')
    #
    # print(float(100 - np.mean(np.abs(y_prediction_train - y_data_train)) * 100))
    # print(float(100 - np.mean(np.abs(y_prediction_test - y_data_test)) * 100))
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
    #     x_data_set, y_data_set, test_size=0.20)
    #
    # theta_vector = np.zeros(((x_data_train.shape[1] + 1), 1), dtype='f')
    #
    # l_t = LogisticTrainer(0.0004)
    # parameters = l_t.trains(x_data_train, y_data_train, theta_vector)
    # y_prediction_test = l_t.classify(x_data_test, parameters)
    # y_prediction_train = l_t.classify(x_data_train, parameters)
    #
    # data = np.hstack((y_data_test, y_prediction_test))
    # output_name = pd.DataFrame(data, columns=['actual_data', 'pred_value'])
    # output_name.to_csv("iris_3.csv", sep=',')
    #
    # print(float(100 - np.mean(np.abs(y_prediction_train - y_data_train)) * 100))
    # print(float(100 - np.mean(np.abs(y_prediction_test - y_data_test)) * 100))

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

    l_rate = 0.0048
    l_t = LogisticTrainer(l_rate)

    parameters = l_t.trains(x_data_train, y_data_train, theta_vector)
    y_prediction_test = l_t.classify(x_data_test, parameters)
    y_prediction_train = l_t.classify(x_data_train, parameters)

    train_acc = round(float(100 - np.mean(np.abs(y_prediction_train - y_data_train)) * 100))
    test_acc = round(float(100 - np.mean(np.abs(y_prediction_test - y_data_test)) * 100))

    print("train accuracy:", train_acc)
    print("test accuracy:", test_acc)

    # train_acc_list = []
    # for value in l_rate:
    #     l_t = LogisticTrainer(value)
    #     parameters = l_t.trains(x_data_train, y_data_train, theta_vector)
    #     y_prediction_test = l_t.classify(x_data_test, parameters)
    #     y_prediction_train = l_t.classify(x_data_train, parameters)
    #     # print("learning rate:", value)
    #     train_acc = round(float(100 - np.mean(np.abs(y_prediction_train - y_data_train)) * 100))
    #     test_acc = round(float(100 - np.mean(np.abs(y_prediction_test - y_data_test)) * 100))
    #     # print("train accuracy:", train_acc)
    #     # print("test accuracy:", test_acc)
    #     train_acc_list.append(train_acc)

    # plt.plot(l_rate, train_acc_list)
    # plt.show()
    # df = pd.read_csv('Immunotherapy.csv')
    #
    # features = ["sex","age","Time","Number_of_Warts","Type","Area","induration_diameter"]
    # predicted_feature = ["Result_of_Treatment"]
    #
    # x_data_set = np.array(pd.DataFrame(df, columns=features))
    # y_data_set = np.array(pd.DataFrame(df, columns=predicted_feature))
    #
    # x_data_set = (x_data_set - x_data_set.mean()) / x_data_set.std()
    #
    # theta_vector = np.zeros(((len(features)+1), 1), dtype='f')
    #
    # x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
    #     x_data_set, y_data_set, test_size=0.35)
    #
    # l_t = LogisticTrainer(0.0001)
    #
    # parameters = l_t.trains(x_data_train, y_data_train, theta_vector)
    # y_prediction_test = l_t.classify(x_data_test, parameters)
    # y_prediction_train = l_t.classify(x_data_train, parameters)
    # train_acc = round(float(100 - np.mean(np.abs(y_prediction_train - y_data_train)) * 100))
    # test_acc = round(float(100 - np.mean(np.abs(y_prediction_test - y_data_test)) * 100))
    # print("train accuracy:", train_acc)
    # print("test accuracy:", test_acc)
    # plt.plot(x_data_set,y_data_set, 'ob')
    # plt.plot(x_data_test, y_prediction_test)
    # plt.show()


if __name__ == '__main__':
    main()

