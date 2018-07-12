import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

df = pd.read_csv('iris.csv')
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
predicted_feature = ['species']

X = np.array(pd.DataFrame(df, columns=features))
Y = np.array(pd.DataFrame(df, columns=predicted_feature))

x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
    X, Y, test_size=0.40, shuffle=True)

k = 3

y_pred = []

for i in range(0, len(x_data_test)):
    a = b = c = 0
    distance = []
    temp = 0
    for j in range(0, len(x_data_train)):
        for k in range(0, x_data_train.shape[1]):
            temp += pow(x_data_train[j][k] - x_data_test[i][k], 2)
        d1 = [math.sqrt(temp), y_data_train[j][0]]
        distance.append(d1)
    distance = sorted(distance)
    distance = distance[0:k]
    for l in range(0, len(distance)):
        if distance[l][1] == "setosa":
            a += 1
        elif distance[l][1] == "versicolor":
            b += 1
        else:
            c += 1
    if a == max(a, b, c):
        y_pred.append('setosa')
    elif b == max(a, b, c):
        y_pred.append('versicolor')
    else:
        y_pred.append('virginica')

print(y_pred)
print(y_data_test)

