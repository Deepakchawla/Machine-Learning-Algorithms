import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

class LinearTrainer:

    def __init__(self):

        # Learning Rate
        self.l_rate = 0.0001

        # Total iterations
        self.iterations = 60000

    def trainNativePython(self, x_data_train, y_data_train, theta_vector):
        x_data_train = np.column_stack((np.ones((x_data_train.shape[0], 1), dtype=float), x_data_train))
        cost = np.zeros(self.iterations)

        for i in range(self.iterations):
            y_pred =  np.dot(x_data_train, theta_vector)
            delta = y_pred -y_data_train
            #dtheta = np.zeros((x_data_train.shape[1],1), dtype=np.float)
            dtheta = np.dot(x_data_train.T, delta)
            # for k in range (x_data_train.shape[0]):
            #     temp = y_pred[k] -y_data_train[k]
            #     temp2[i] += temp*temp
            #     for j in range (x_data_train.shape[1]):
            #         x = x_data_train[k,j]
            #         dtheta[j] += (x * temp)

           # print (temp3 == dtheta)
            cost[i] = np.dot(delta.T, delta)
            cost[i] /= (2* len(x_data_train))
            dtheta = ((dtheta * self.l_rate) / len(x_data_train))
            theta_vector = theta_vector - dtheta
            if (i %500 == 0):
                print ("Iteration=", i, "Cost = ", cost[i])

        # self.plotgraph(np.arange(self.iterations), '1 ', temp2)

        return theta_vector

    def classify(self, x_data_test, theta_vector):

        x_data_test = np.column_stack((np.ones((x_data_test.shape[0], 1)), x_data_test))

        return np.dot(x_data_test, theta_vector)

    def printDiagnostics(self, y_data, y_pred):
        y_dataPredErr = np.zeros(y_pred.shape, dtype=np.float)
        errRanges = np.arange(0.05,2,0.05)
        prevSumRange = 0
        for i in range(len(errRanges)):
            y_dataPredErr [(y_pred - y_data)/y_data <= errRanges[i] ] = errRanges[i]
            sumRange = np.sum(y_dataPredErr == errRanges[i])
            print ("Num With Error Range", errRanges[i], " =", sumRange - prevSumRange, sumRange)
            prevSumRange =  sumRange
            #y_dataPredErr = np.zeros(y_pred.shape, dtype=np.float)

        y_dataPredErr [(y_pred - y_data)/y_data >  errRanges[i] ] = errRanges[i] + 0.05
        Others = np.sum(y_dataPredErr == errRanges[i]+0.05)
        print("Total Data set diagnosed=", sumRange + Others, "Data Set Avail=", y_pred.shape[0], "\n\n")


    def accuracy(self, y_data, y_pred):

        total_error = 0
        for i in range(0, len(y_data)):
            total_error += abs((y_pred[i] - y_data[i]) / y_data[i])
        total_error = (total_error / len(y_data))
        accuracy = 1 - total_error
        self.printDiagnostics(y_data,y_pred)
        return accuracy * 100

    def plotgraph(self, x_data_test, y_data_test, y_pred):

        # plt.scatter(x_data_test, y_data_test, color='g', label='Test Data Set')
        plt.plot(x_data_test, y_pred, color='r', label='Predicted Values')
        plt.legend()
        plt.show()


def main():

    df = pd.read_csv('final_data.csv')
    features = ['bathrooms', 'bedrooms', 'finishedsqft'] #, 'totalrooms']
    predicted_feature = ['price']

    df.loc[:, "bathrooms"] = (df.loc[:, "bathrooms"] - df.loc[:, "bathrooms"].mean())/df.loc[:, "bathrooms"].std()
    df.loc[:, "bedrooms"]= (df.loc[:, "bedrooms"] - df.loc[:, "bedrooms"].mean())/df.loc[:, "bedrooms"].std()
    df.loc[:, "finishedsqft"] = (df.loc[:, "finishedsqft"] - df.loc[:, "finishedsqft"].mean())/df.loc[:, "finishedsqft"].std()

    x_data_set = np.array(pd.DataFrame(df, columns=features))
    y_data_set = np.array(pd.DataFrame(df, columns=predicted_feature))
    print ("x_data_set.size, shape",  x_data_set.size, x_data_set.shape)
    print ("y_data_set.size, shape",  y_data_set.size, y_data_set.shape)
    l_t = LinearTrainer()
    theta_vector = np.zeros(((len(features)+1), 1), dtype='f')
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        x_data_set, y_data_set, test_size=0.10, shuffle=False)

    print ("x_data_train.size, shape",  x_data_train.size, x_data_train.shape)
    print ("x_data_test.size, shape",  x_data_test.size, x_data_test.shape)
    print ("y_data_train.size, shape",  y_data_set.size, y_data_train.shape)
    print ("y_data_test.size, shape",  y_data_test.size, y_data_test.shape)


    t1 =time.time()
    parameters = l_t.trainNativePython(x_data_train, y_data_train, theta_vector)
    t2 =time.time()
    print ("Training time = ", t2-t1)
    y_prediction_test = l_t.classify(x_data_test, parameters)
    y_prediction_train = l_t.classify(x_data_train, parameters)

#    dtype = [('Pred_train', 'float32'), ('Actual_train', 'float32'), ('Pred_test', 'float32'), ('Actual_test', 'float32')]
    colsTrain = ['Pred_train', 'Actual_train']
    colsTest = ['Pred_test',  'Actual_test']
    y_pandas_train = np.hstack((y_prediction_train , y_data_train))
    y_pandas_test = np.hstack((y_prediction_test , y_data_test))
    output_pandas_train = pd.DataFrame(y_pandas_train, columns = colsTrain)
    output_pandas_train.to_csv("finalTrain_out.csv", sep = ',')
    output_pandas_test = pd.DataFrame(y_pandas_test, columns = colsTest)
    output_pandas_test.to_csv("finalTest_out.csv", sep = ',')

    accuracy_test = l_t.accuracy(y_data_test, y_prediction_test)
    accuracy_train = l_t.accuracy(y_data_train, y_prediction_train)

    print("accuracy_test=", accuracy_test, " accuracy_train=", accuracy_train)
    print(mean_squared_error(y_data_test, y_prediction_test))
    print(np.sqrt(mean_squared_error(y_data_test, y_prediction_test)))
    # l_t.plotgraph(x_data_test, y_data_test, y_prediction)
    print(parameters)


if __name__ == '__main__':
    main()