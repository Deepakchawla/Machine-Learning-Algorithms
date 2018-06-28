import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class LinearTrainer:

    def __init__(self):

        # Learning Rate
        self.l_rate = 0.0001

        # Total iterations
        self.iterations = 60000

    def trains(self, x_data_train, y_data_train, theta_vector):
        x_data_train = np.column_stack((np.ones((x_data_train.shape[0], 1), dtype=float), x_data_train))
        temp2 = np.zeros(self.iterations)

        for i in range(self.iterations):
            temp = ((np.dot(x_data_train, theta_vector)) - y_data_train)
            temp1 = np.power((np.dot(x_data_train, theta_vector) - y_data_train), 2)
            temp2[i] = np.sum(temp1) / 2 * len(x_data_train)
            temp = np.dot(np.transpose(x_data_train), temp)
            temp = ((temp * self.l_rate) / len(x_data_train))
            theta_vector = theta_vector - temp

        # self.plotgraph(np.arange(self.iterations), '1 ', temp2)

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

        # plt.scatter(x_data_test, y_data_test, color='g', label='Test Data Set')
        plt.plot(x_data_test, y_pred, color='r', label='Predicted Values')
        plt.legend()
        plt.show()


def main():

    # df = pd.read_csv('Housing.csv')
    # features = ['bathrooms', 'bedrooms', 'finishedsqft', 'totalrooms']
    # predicted_feature = ['price']

    # df = pd.read_csv('iris.csv', usecols=range(0,4))
    # features = ['sepal_length', 'sepal_width', 'petal_width']
    # predicted_feature = ['petal_length']

    df = pd.read_csv('petrol_consumption.csv')
    features = ['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']
    predicted_feature = ['Petrol_Consumption']

    df = (df - df.mean()) / df.std()

    x_data_set = np.array(pd.DataFrame(df, columns=features))
    y_data_set = np.array(pd.DataFrame(df, columns=predicted_feature))

    l_t = LinearTrainer()
    theta_vector = np.zeros(((len(features)+1), 1), dtype='f')
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        x_data_set, y_data_set, test_size=0.25, shuffle=False)

    parameters = l_t.trains(x_data_train, y_data_train, theta_vector)
    y_prediction = l_t.classify(x_data_test, parameters)
    accuracy = l_t.accuracy(y_data_test, y_prediction)

    print(accuracy)
    print(mean_squared_error(y_data_test, y_prediction))
    print(np.sqrt(mean_squared_error(y_data_test, y_prediction)))
    # l_t.plotgraph(x_data_test, y_data_test, y_prediction)


if __name__ == '__main__':
    main()
