# import necessary modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split


df = pd.read_csv('iris.csv')

# Assign the split data into train and test variables.
x_data_set = np.array(pd.DataFrame(df, columns=['sepal_length'])['sepal_length'])
y_data_set = np.array(pd.DataFrame(df, columns=['sepal_width'])['sepal_width'])

x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        x_data_set, y_data_set, test_size=0.25, shuffle=False)

# Learning Rate
l_rate = 0.0001

# Total iterations
iterations = 2000

# length of the train and test datasets.
m = len(x_data_train)
n = len(y_data_test)

# Function gradient_descent() is calculate the theta0 and theta1 for hypothesis of linear regression.
def gradient_descent():
    temp1 = temp2 = i = 0
    newtheta0 = 0
    newtheta1 = 1
    y_pred = np.array(0)

    # iterations loop then assign theta0 and theta1 to newtheta0 and newtheta1 variables.
    while i < iterations:
        for j in range(m):
            temp1 += ((newtheta0 + newtheta1 * x_data_train[j]) - y_data_train[j])
            temp2 += ((newtheta0 + newtheta1 * x_data_train[j]) - y_data_train[j]) * x_data_train[j]
        newtheta0 = newtheta0 - ((l_rate * temp1)/m)
        newtheta1 = newtheta1 - ((l_rate * temp2)/m)
        i += 1
    # predict the values by giving x_input_test.
    for i in range(len(x_data_test)):
        temp = (newtheta0 + newtheta1 * x_data_test[i])
        temp = float(str(temp)[0:3])
        y_pred = np.append(y_pred, temp)

    # call error_calculate()
    print("Accuracy in model is",  avgerror(np.asarray(y_pred)), type(y_pred))
    # graph(y_pred)


# Function error_calculate() is calculate the accuracy of the predicted values with the input y test values.
def avgerror(y_pred_test):
    total_error = 0
    for i in range(0, n):
        total_error += abs((y_pred_test[i] - y_data_test[i])/y_data_test[i])
    total_error = (total_error/len(y_data_test))
    accuracy = 1 - total_error
    return accuracy * 100


def graph(y_pred):
    plt.scatter(x_data_test, y_data_test, color='g', label='whole data')
    plt.plot(x_data_test, y_pred, color='r', label='predicted value')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Call gradient_descent() function to the build the model using learning algorithm.
    gradient_descent()
