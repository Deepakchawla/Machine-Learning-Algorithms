import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


class LinearTrainer:

    def __init__(self):

        # Learning Rate
        self.l_rate = 0.0001

        # Total iterations
        self.iterations = 60000

    def trains(self, x_data_train, y_data_train, theta_vector):
        i = 0

        x_data_train = np.column_stack((np.ones((x_data_train.shape[0], 1)), x_data_train))

        while i <= self.iterations:
            temp = ((np.dot(x_data_train, theta_vector)) - y_data_train)
            temp = np.dot(np.transpose(x_data_train), temp)
            temp = ((temp * self.l_rate) / len(x_data_train))
            theta_vector = theta_vector - temp
            i += 1

        return theta_vector

    def classify(self, x_data_test, theta_vector):

        x_data_test = np.column_stack((np.ones((x_data_test.shape[0], 1)), x_data_test))
        return np.dot(x_data_test, theta_vector)

    def accuracy(self, y_data_test, y_pred_test):

        total_error = 0
        for i in range(0, len(y_data_test)):
            total_error += abs((y_pred_test[i] - y_data_test[i]) / y_data_test[i])
        total_error = (total_error / len(y_data_test))
        accuracy = 1 - total_error
        return accuracy * 100

    def plotgraph(self, x_data_test, y_data_test, y_pred):

        plt.scatter(x_data_test, y_data_test, color='g', label='Test Data Set')
        plt.plot(x_data_test, y_pred, color='r', label='Predicted Values')
        plt.legend()
        plt.show()


def main():

    df = pd.read_csv('iris.csv')

    features = ['sepal_length', 'petal_width', 'sepal_width']
    predicted_feature = ['petal_length']

    # Assign the split data into train and test variables.
    x_data_set = np.array(pd.DataFrame(df, columns=features))
    y_data_set = np.array(pd.DataFrame(df, columns=predicted_feature))

    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        x_data_set, y_data_set, test_size=0.25, shuffle=False)
    theta_vector = np.zeros(((len(features)+1), 1), dtype='f')

    l_t = LinearTrainer()
    parameters = l_t.trains(x_data_train, y_data_train,theta_vector)
    y_prediction = l_t.classify(x_data_test, parameters)
    accuracy = l_t.accuracy(y_data_test, y_prediction)
    print(accuracy)
    # l_t.plotgraph(x_data_test, y_data_test, y_prediction)


if __name__ == '__main__':
    main()
