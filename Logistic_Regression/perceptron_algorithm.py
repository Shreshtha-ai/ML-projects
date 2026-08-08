from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
X,y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0, n_classes=2, 
                        n_clusters_per_class=1, random_state=42,hypercube=False,class_sep=1.3) 
plt.figure(figsize = (10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter', s=100 )
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Algorithm")
plt.show()    