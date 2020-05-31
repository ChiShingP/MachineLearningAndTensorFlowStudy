import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

import matplotlib.pyplot as pyplot
import pickle 
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#Label
predict = "G3" 
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#splits data (10%) into 4 arrays 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

##train model until accuracy > 95%
#best = 0
#for _ in range(30):
#    #Linear Regression: Finds best fit line from scattered plot, This is used on data that has correlation on each other
#    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
#    linear = linear_model.LinearRegression()    #Training Model
#    linear.fit(x_train, y_train)                #finds best fit line using training data (?)
#    acc = linear.score(x_test, y_test)          #accuracy of model when used on test data
#    print('Accuracy: ', acc)

#    #If current score is the best version, then save it in a pickle file.
#    if acc > best:
#        best = acc
#        #saving models
#        with open("studentmodel.pickle", "wb") as f:
#            pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)  #Correlation between each independent variable, higher = more correlated
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)
print('Predicted Values\t Data Set\t\tActual Value')
for x in range(len(predictions)):
    print(predictions[x],'\t', x_test[x],'\t', y_test[x])

#Plotting the correlation between p and the final grade
p = 'studytime'    #Variable in relation to final grade
style.use("ggplot")
pyplot.scatter(data[p], data["G3"]) #p vs final grade
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()