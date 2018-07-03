from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# df = pd.read_csv('iris.csv')
# features = ['sepal_length', 'sepal_width', 'petal_width']
# predicted_feature = ['petal_length']

# df = pd.read_csv('petrol_consumption.csv')
# features = ['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']
# predicted_feature = ['Petrol_Consumption']


df = pd.read_csv('final_data.csv')
features = ['bathrooms', 'bedrooms', 'finishedsqft', 'totalrooms']
predicted_feature = ['price']


X = pd.DataFrame(df, columns=features)
Y = pd.DataFrame(df, columns=predicted_feature)


x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
    X, Y, test_size=0.20, random_state=0)

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

print(mean_squared_error(y_data_test, y_pred_test), np.sqrt(mean_squared_error(y_data_test, y_pred_test)), accuracy*100)
