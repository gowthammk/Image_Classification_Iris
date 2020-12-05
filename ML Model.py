import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("R:\\Web Based Iris dataset\\archive\\Iris.csv")
y = data["Species"]
y_lab = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
y = y.apply(lambda x: y_lab.index(x))
x = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

clf=RandomForestClassifier(n_estimators=100).fit(X_train,y_train)


pickle.dump(clf, open("iris.pkl", "wb"))