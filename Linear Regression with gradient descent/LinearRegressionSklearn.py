from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# df = pd.read_csv('iris.csv')
# features = ['sepal_length', 'sepal_width', 'petal_width']
# predicted_feature = ['petal_length']

# df = pd.read_csv('petrol_consumption.csv')
# features = ['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']
# predicted_feature = ['Petrol_Consumption']


# df = pd.read_csv('final_data.csv')
# features = ['bathrooms', 'bedrooms', 'finishedsqft', 'totalrooms']
# predicted_feature = ['price']

df = pd.read_csv('airfoil_self_noise.dat.txt', delimiter='\t')
df.columns = ['Frequency', 'AngleOfAttack', 'ChordLength', 'FreeStreamVelocity', 'SuctionSideDisplacementThickness', 'ScaledSoundPressureLevel']
features = ['Frequency', 'AngleOfAttack', 'ChordLength', 'FreeStreamVelocity', 'SuctionSideDisplacementThickness']
predicted_feature = ['ScaledSoundPressureLevel']


X = pd.DataFrame(df, columns=features)
Y = pd.DataFrame(df, columns=predicted_feature)


x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
    X, Y, test_size=0.20, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(x_data_train, y_data_train)

y_pred_test = regr.predict(x_data_test)
y_pred_train = regr.predict(x_data_train)

y_pred_test = np.array(y_pred_test).flatten()
y_data_test = np.array(y_data_test).flatten()
y_data_train = np.array(y_data_train).flatten()
y_pred_train = np.array(y_pred_train).flatten()


def accuracy(Y_data, Y_pred):

    total_error = 0
    for i in range(0, len(Y_data)):
        total_error += abs((Y_pred[i] - Y_data[i]) / Y_data[i])
    total_error = (total_error / len(Y_data))
    return (1 - total_error) * 100


print("Test MSE: ", mean_squared_error(y_data_test, y_pred_test), "Train MSE: ", mean_squared_error(y_data_train, y_pred_train))
print("accuracy train: ", accuracy(y_data_train, y_pred_train), "accuracy test: ", accuracy(y_data_test, y_pred_test))
