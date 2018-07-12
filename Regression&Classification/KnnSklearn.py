from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('iris.csv')
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
predicted_feature = ['species']

# X = pd.DataFrame(df, columns=features)
# Y = pd.DataFrame(df, columns=predicted_feature)
X = df.iloc[:, :-1].values
Y = df.iloc[:, 4].values

x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
    X, Y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(x_data_train)

x_data_train = scaler.transform(x_data_train)
x_data_test = scaler.transform(x_data_test)

clg = KNeighborsClassifier(n_neighbors=5)
clg.fit(x_data_train, y_data_train)
y_pred_test = clg.predict(x_data_test)

print(confusion_matrix(y_data_test, y_pred_test))
print(classification_report(y_data_test, y_pred_test))