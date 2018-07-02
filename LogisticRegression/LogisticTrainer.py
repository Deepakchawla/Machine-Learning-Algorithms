import numpy as np
import h5py


class LogisticTrainer:

    def __init__(self):

        # Learning Rate
        self.l_rate = 0.0001

        # Total iterations
        self.iterations = 2000

    def trains(self, x_data_train, y_data_train, theta_vector):

        x_data_train = np.column_stack((np.ones((x_data_train.shape[0], 1), dtype=float), x_data_train))
        for i in range(self.iterations):
            temp = (np.dot(x_data_train, theta_vector) - y_data_train)
            temp = np.dot(np.transpose(x_data_train), temp)
            temp = ((temp * self.l_rate) / len(x_data_train))
            theta_vector = theta_vector - temp
        return theta_vector

    def classify(self, x_data_test, theta_vector):

        y_prediction = np.zeros((x_data_test.shape[0], 1), dtype=float)
        x_data_test = np.column_stack((np.ones((x_data_test.shape[0], 1)), x_data_test))
        temp = np.dot(x_data_test, theta_vector)
        A = np.array(1 / (1 + np.exp(-(temp))))

        for i in (range(0, len(A))):
            if round(A[i][0],2) <= 0.50:
                y_prediction[i][0] = 0
            else:
                y_prediction[i][0] = 1

        return y_prediction

    def accuracy(self, y_data, y_pred):

        return 100 - np.mean(np.abs(y_pred - y_data)) * 100


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

    theta_vector = np.zeros(((x_data_train.shape[1]+1), 1), dtype='f')

    l_t = LogisticTrainer()
    parameters = l_t.trains(x_data_train, y_data_train, theta_vector)
    y_prediction_test = l_t.classify(x_data_test, parameters)
    y_prediction_train = l_t.classify(x_data_train, parameters)

    print("Train accuracy:", l_t.accuracy(y_data_train, y_prediction_train), "%")
    print("Test accuracy:", l_t.accuracy(y_data_test, y_prediction_test), "%")


if __name__ == '__main__':
    main()
