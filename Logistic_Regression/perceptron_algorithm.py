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

def perceptron(X,y):
    
    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1
    
    for i in range(1000):
        j = np.random.randint(0,100)
        y_hat = step(np.dot(X[j],weights)) # y = w0 + w1*x1 + w2*x2
        weights = weights + lr*(y[j]-y_hat)*X[j]
        
    return weights[0],weights[1:]
    
def step(z):
    return 1 if z>=0 else 0 # this is the activation function we use in the perceptron algorithm


intercept_,coef_ = perceptron(X,y)