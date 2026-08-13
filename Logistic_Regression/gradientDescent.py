from sklearn.datasets import make_classification
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X,y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0, n_classes=2, 
                        n_clusters_per_class=1, random_state=42,hypercube=False,class_sep=2) 
plt.figure(figsize = (10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter', s=100 )
plt.show()

lor = LogisticRegression(penalty=None,solver='sag') #sag means Stochastic Gradient Descent
lor.fit(X,y)

print(lor.coef_)
print(lor.intercept_)

m1 = -(lor.coef_[0][0]/lor.coef_[0][1])
b1 = -(lor.intercept_/lor.coef_[0][1])

x_input = np.linspace(-3,3,100)
y_input = m1*x_input + b1

def gd(X,y):
    X=np.insert(X,0,1,axis=1)
    weights=np.ones(X.shape[1])
    lr=0.01
    

