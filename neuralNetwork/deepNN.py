import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score
import h5py


class LinearTrainer:

    def __init__(self):

        # Learning Rate
        self.l_rate = 0.001
        # Total iterations
        self.iterations = 6000

    def trains(self, x_data_train, y_data_train):
        n = [x_data_train.shape[1], 30,50,40,20,1]
        i = 0
        w = []
        b = []
        z = [0] * 6
        a = [0] * 6
        g = [0] * 6
        da = [0] * 6
        dz = [0] * 6
        dw = [0] * 6
        db = [0] * 6
        a[0] = x_data_train.T
        m = x_data_train.shape[0]
        for k in range(1, 6):
            w.append(np.random.rand(n[k], n[k-1]) * 0.01)
            b.append(np.zeros((n[k], 1), dtype='float'))
        while i<self.iterations:
            j = 0
            while j < 5:
                z[j] = np.dot(w[j], a[j]) + b[j]
                a[j+1] = 1 / 1 + np.exp(-z[j])
                g[j] = (1 / (1 + np.exp(-z[j]))) * (1 - (1 / (1 + np.exp(-z[j]))))
                da[j] = (-(y_data_train.T / a[j+1]) + ((1 - y_data_train.T) / (1 - a[j+1])))
                dz[j] = da[j] * g[j]
                dw[j] = np.dot(dz[j], (np.transpose(a[j])))/m
                db[j] = np.sum(dz[j], axis=1, keepdims=True) / m
                w[j] = w[j] - np.dot(self.l_rate, dw[j])
                b[j] = b[j] - np.dot(self.l_rate, db[j])
                j+=1
            i+=1
        return w, b

    def classify(self, x_data_test, parameters):
        j = 0
        z = [0] * 6
        a = [0] * 6
        a[0] = x_data_test.T

        while j < 5:
            z[j] = np.dot(parameters[0][j], a[j]) + parameters[1][j]
            a[j+1] = 1 / 1 + np.exp(-z[j])
            j+=1

        return a[-1]

    def accuracy(self, y_data_test, y_pred_test):
        y_pred_test =y_pred_test[0
        ]
        total_error = 0
        error = []
        for i in range(0, len(y_data_test)):
            total_error = total_error + abs((y_pred_test[i] - y_data_test[i]) / y_data_test[i])
            error.append(total_error)
        total_error = (total_error / len(y_data_test))
        accuracy = 1 - total_error
        return [accuracy * 100, error]


def main():


    df = pd.read_csv('iris.csv')

    # Assign the split data into train and test variables.
    x_data_set = np.array(pd.DataFrame(df, columns=['sepal_length', 'petal_length', 'petal_width']))
    y_data_set = np.array(pd.DataFrame(df, columns=['sepal_width']))

    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        x_data_set, y_data_set, test_size=0.30, random_state=0)
    l_t = LinearTrainer()
    parameters = l_t.trains(x_data_train, y_data_train)
    y_prediction = l_t.classify(x_data_test, parameters)
    y_prediction_train = l_t.classify(x_data_train, parameters)
    accuracy = l_t.accuracy(y_data_test, y_prediction)
    accuracy_train = l_t.accuracy(y_data_train, y_prediction_train)

    print("test: ", accuracy[0], "train: ", accuracy_train[0])

if __name__ == '__main__':
    main()
