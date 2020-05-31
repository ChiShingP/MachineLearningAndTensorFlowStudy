#K-Nearest Neighbors, K = # of groups and should be an odd number in order to avoid 2 vs 2 = tie
#Given an irregulate data point, classify it into a category by determining how close it is to other groups of data
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing



data = pd.read_csv("car.data")
print(data.head())

#Turning each column into its own list as an integer instead of string
le  = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))       #Feature
Y = list(cls)                                                       #Label

#splits data (10%) into 4 arrays 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)


model = KNeighborsClassifier(5) #HyperVariable: 5 // Variable that has to be adjusted as we train the model
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

names = ["unacc", "acc", "good", "VeryGood"] #0, 1, 2, 3
predicted = model.predict(x_test)
for x in range (len(predicted)):
    print("Predicted: ", names[predicted[x]], "\tData: ", x_test[x], "\tActual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("\nN: ", n)       #returns the distance between indexes, and the index of the points