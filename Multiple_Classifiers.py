# Create a Machine Learning Classification Application
# Use SciKit for Machine Learning
from sklearn import datasets, tree, metrics
from sklearn.neighbors import KNeighborsClassifier
# Use a Decision tree classifier.

# Use KNN classifier.
# Import and use a larger existing data model
# Understand what makes a feature choice good and bad.
# Understand how to use more features in your data model.
# Understand how to classify more than two types.
# Understand how to use more than one classier
# Understand how to compare classifiers.

#Dataset
iris = datasets.load_iris()

x = iris.data

y = iris.data

#create a training set
from sklearn.model_selection import train_test_split
#training x
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

# print(x_train)
# print(y_train)
#import decision tree
clf = tree.DecisionTreeClassifier()


#Training the Classifier
clf.fit(x_train, y_train)

#prediction
prediction = tree.DecisionTreeClassifier.predict(x_test, y_test)


#printing the predictions
KNN_CLF = KNeighborsClassifier()
KNN_CLF(x_train, y_train)
print("Printing our Predictions: ")
print(prediction)
print(metrics.accuracy_score(y_test, prediction, normalize=False))

print(metrics.accuracy_score(y_test, prediction, KNN_CLF))
