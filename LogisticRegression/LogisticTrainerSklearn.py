from sklearn.linear_model import LogisticRegression
import h5py
import numpy as np

trainfile = 'train_catvnoncat.h5'
testfile = 'test_catvnoncat.h5'

train_dataset = h5py.File(trainfile, "r")
test_dataset = h5py.File(testfile, "r")

x_data_train = np.transpose(np.reshape(np.array(train_dataset['train_set_x'][:]), (np.array(train_dataset['train_set_x'][:]).shape[0], -1)))
y_data_train = (np.array([train_dataset['train_set_y'][:]]))

x_data_test = np.transpose(np.reshape(np.array(test_dataset['test_set_x'][:]), (np.array(test_dataset['test_set_x'][:]).shape[0], -1)))
y_data_test = (np.array([test_dataset['test_set_y'][:]]))

x_data_train = x_data_train/255.
x_data_test = x_data_test/255.

lr = LogisticRegression(C=1000.0, random_state=0)

lr.fit(x_data_train.T, y_data_train.T.ravel())

Y_prediction = lr.predict(x_data_test.T)
Y_prediction_train = lr.predict(x_data_train.T)

print(lr.coef_)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - y_data_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - y_data_test)) * 100))