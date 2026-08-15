import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_decision_regions

iris = load_iris()

# Extract features and labels
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

df = df[['sepal length (cm)', 'petal length (cm)', 'target']]

print(df.head())

X = df.iloc[:,0:2]  # features it contains only first two columns (SL AND PL)
y = df.iloc[:,2]  # target it contains only last column (TARGET)

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=2)

clf = LogisticRegression()  #before i was using softmax here it was clf = LogisticRegression(multi_class='multinomial') but it is deprecated now 
clf.fit(X_train, y_train) # training the model

y_pred = clf.predict(X_test)
print(y_pred)

print(accuracy_score(y_pred,y_test)) # 0.9666666666666667

print(pd.DataFrame(confusion_matrix(y_test,y_pred)))

#prediction 

query = pd.DataFrame([[4.3,2.1]], columns=['sepal length (cm)', 'petal length (cm)'])
print(clf.predict_proba(query)) 
print(clf.predict(query)) #so our query is setosa i.e 0

plot_decision_regions(X.values, y.values, clf,legend =2) #here X & y must be in numpy array format otherwise we will get error
plt.xlabel("sepal length (cm)")
plt.ylabel("petal length (cm)")
plt.title("Softmax Regression Decision Regions")
plt.show()


