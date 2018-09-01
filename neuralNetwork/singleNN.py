import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score
import h5py

class LinearTrainer:

    def __init__(self):

        # Learning Rate
        self.l_rate = 0.0004
        # Total iterations
        self.iterations = 2000

    def trains(self, x_data_train, y_data_train):
        i =0
        m = x_data_train.shape[1]
        x_data_train = np.transpose(x_data_train)
        y_data_train = np.transpose(y_data_train)
        w = np.random.rand(x_data_train.shape[0], 1) * 0.01
        b = np.zeros((1, 1), dtype='float')

        while i<self.iterations:

            z = np.dot(w.T, x_data_train) + b
            a = 1 / 1 + np.exp(-z)

            dz = a - y_data_train
            dw = np.dot(x_data_train, dz.T) / m
            db = dz.sum() / m
            w = w - np.dot(self.l_rate, dw)
            b = b - np.dot(self.l_rate, db)
            i+=1

        return w,b

    def classify(self, x_data_test, parameters):
        z =  np.dot(parameters[0].T, x_data_test.T) + parameters[1]
        return 1 / 1 + np.exp(-z)

    def accuracy(self, y_data_test, y_pred_test):

        total_error = 0
        error = []
        for i in range(0, len(y_data_test)):
            total_error = total_error + abs((y_pred_test[i] - y_data_test[i]) / y_data_test[i])
            error.append(total_error)
        total_error = (total_error / len(y_data_test))
        accuracy = 1 - total_error
        return [accuracy * 100, error]

    def plotgraph(self, x_data_test, y_data_test, y_pred):
        plt.plot(x_data_test, y_data_test, 'or', label='whole data')
        plt.plot(x_data_test, y_pred, label='predicted value')
        plt.legend()
        plt.show()


def main():

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

    l_t = LinearTrainer()
    parameters = l_t.trains(x_data_train, y_data_train)
    y_prediction_train = l_t.classify(x_data_train, parameters)
    y_prediction_test = l_t.classify(x_data_test, parameters)

    train_acc = round(float(100 - np.mean(np.abs(y_prediction_train - y_data_train)) * 100))

    test_acc = round(float(100 - np.mean(np.abs(y_prediction_test - y_data_test)) * 100))
    #
    print("train accuracy:", train_acc)
    print("test accuracy:", test_acc)
    #
    # print(accuracy[0], accuracy_train[0])
    # plt.plot(accuracy[1],'b')
    # plt.plot(accuracy_train[1], 'g')
    # plt.show()
    # plt.plot(x_data_test, y_data_test, 'og', label='whole data')
    # plt.plot(x_data_test, y_prediction, label='predicted value')
    # plt.legend()
    # plt.show()

    # l_t.plotgraph(x_data_test, y_data_test, y_prediction)
    # print(np.sqrt(mean_squared_error(y_data_train, y_prediction_train)))
    # print(np.sqrt(mean_absolute_error(y_data_test, y_prediction)))


if __name__ == '__main__':
    main()
