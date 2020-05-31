#support vector machine, the larger the margin the more accuracte the result.
#S
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

#clf = KNeightborsClassifier(10)
clf = svm.SVC(kernel = "poly", degree = 5)  #linear/ poly
clf.fit(x_train, y_train)

y_prediction = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_prediction)
print(acc)