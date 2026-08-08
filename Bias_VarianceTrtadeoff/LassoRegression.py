#generating random data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

#1. GENERATE NOISY QUADRATIC DATA
X = 5* np.random.rand(100,1) #generating 100 random values between 0 and 5
y = 0.7*X**2+ 2*X + 5 + np.random.randn(100,1) #adding some noise 

plt.scatter(X,y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Random Data Generation")
plt.savefig(fname="data.png") #save the plot 
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X.reshape(100,1), y.reshape(100), test_size=0.2, random_state=42) #split the data into training and testing data

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=10) #it is used for to generate polynomial features 
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test) # we never use fit_transform() on test data only use transform because the training data already contains all the feature

from mlxtend.evaluate import bias_variance_decomp
alphas = np.linspace(0.01,30,100)
loss = []
bias =[]
variance=[]

for i in alphas:
    reg = Lasso(alpha=i)
    avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(
        reg,X_train_poly,y_train,
        X_test_poly,y_test,
        loss = 'mse',
        random_seed=42)
    loss.append(avg_expected_loss)
    bias.append(avg_bias)
    variance.append(avg_variance)

plt.plot(alphas,loss, label = "Expected Loss")
plt.plot(alphas,bias, label = "Bias")
plt.plot(alphas,variance, label="Variance")
plt.xlabel('Alpha')
plt.ylabel('Performance Metric')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.savefig(fname="tradeoff.png")
plt.show()
    

    
        


    

    
    

