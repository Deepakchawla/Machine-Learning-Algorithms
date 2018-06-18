import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


class LinearTrainer:

    def __init__(self):
        # Learning Rate
        self.l_rate = 0.0001
        # Total iterations
        self.iterations = 2000

    def trains(self, x_data_train, y_data_train):

        temp1 = temp2 = i = 0
        theta0 = 0
        theta1 = 1
        # length of the train and test datasets.
        m = len(x_data_train)

        # iterations loop then assign theta0 and theta1 to newtheta0 and newtheta1 variables.
        while i < self.iterations:
            for j in range(m):
                temp1 += ((theta0 + theta1 * x_data_train[j]) - y_data_train[j])
                temp2 += ((theta0 + theta1 * x_data_train[j]) - y_data_train[j]) * x_data_train[j]
            theta0 = theta0 - ((self.l_rate * temp1) / m)
            theta1 = theta1 - ((self.l_rate * temp2) / m)
            i += 1

        return np.array([theta0, theta1])

    def classify(self, x_data_test, parameters):
        x_data_test = np.reshape(x_data_test, (x_data_test.shape[0], 1))
        x_data_test = np.column_stack((np.ones((x_data_test.shape[0], 1)), x_data_test))
        parameters = np.reshape(parameters, (parameters.shape[0], 1))
        return np.dot(x_data_test, parameters)

    def accuracy(self, y_data_test, y_pred_test):
        n = len(y_data_test)
        total_error = 0
        for i in range(0, n):
            total_error += abs((y_pred_test[i] - y_data_test[i]) / y_data_test[i])
        total_error = (total_error / len(y_data_test))
        accuracy = 1 - total_error
        return accuracy * 100

    def plotgraph(self, x_data_test, y_data_test, y_pred):
        plt.scatter(x_data_test, y_data_test, color='g', label='whole data')
        plt.plot(x_data_test, y_pred, color='r', label='predicted value')
        plt.legend()
        plt.show()


def main():
    df = pd.read_csv('iris.csv')

    # Assign the split data into train and test variables.
    x_data_set = np.array(pd.DataFrame(df, columns=['sepal_length'])['sepal_length'])
    y_data_set = np.array(pd.DataFrame(df, columns=['sepal_width'])['sepal_width'])

    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        x_data_set, y_data_set, test_size=0.25, shuffle=False)

    l_t = LinearTrainer()
    parameters = l_t.trains(x_data_train, y_data_train)
    y_prediction = l_t.classify(x_data_test, parameters)
    accuracy = l_t.accuracy(y_data_test, y_prediction)
    print(accuracy)
    l_t.plotgraph(x_data_test, y_data_test, y_prediction)


if __name__ == '__main__':
    main()
