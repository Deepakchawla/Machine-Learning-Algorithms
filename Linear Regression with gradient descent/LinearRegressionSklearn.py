from sklearn import datasets, linear_model
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib import style as sy

sy.use('ggplot')

import pandas as pd
import numpy as np

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
data1 = pd.DataFrame(data= np.c_[iris['data']],
                     columns= iris['feature_names'])

X = pd.DataFrame(data1, columns=['sepal length (cm)'])
Y = pd.DataFrame(data1, columns=['sepal width (cm)'])

x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
    X, Y, test_size=0.20, random_state=42)

regr = linear_model.LinearRegression()
regr.fit(x_data_train, y_data_train)

y_pred_test = regr.predict(x_data_test)
y_pred_test = np.array(y_pred_test).flatten()
y_data_test = np.array(y_data_test).flatten()

total_error = 0
for i in range(0, len(y_data_test)):
    total_error += abs((y_pred_test[i] - y_data_test[i])/y_data_test[i])
total_error = (total_error/len(y_data_test))
accuracy = 1 - total_error

plt.scatter(x_data_test,y_data_test, color='g', label='whole data')
plt.plot(x_data_test, y_pred_test, color='r', label='predicted value')
plt.show()


print(total_error * 100, accuracy*100)
print(explained_variance_score(y_data_test, y_pred_test, multioutput='uniform_average'))
print(r2_score(y_data_test,y_pred_test,multioutput='uniform_average'))