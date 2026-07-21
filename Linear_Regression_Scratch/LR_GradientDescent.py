from Generate_SampleData import X,y, TRUE_B,TRUE_W
class LinearRegression:
    def __init__(self, learning_rate = 0.01): #init initialize the object 
        self.w = 0.0
        self.b = 0.0
        self.lr = learning_rate
        self.cost_history = []
    
    def predict(self, X):
        return [self.w * x + self.b for x in X] 

    def compute_cost(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        cost = sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n
        return cost #this calculate mean squared error 
    
    def compute_gradient(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        dw = (2 / n) * sum((pred - actual) * x for pred, actual, x in zip(predictions, y, X))
        db = (2 / n) * sum(pred - actual for pred, actual in zip(predictions, y)) #bias doesn't depend on x
        return dw, db

    def fit(self, X, y, epochs=1000, print_every=200): #this is used for training the model 
        for epoch in range(epochs):
            dw, db = self.compute_gradient(X, y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0: #print when epoch = 200,400,600....
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f} | w: {self.w:.4f} | b: {self.b:.4f}")
        return self

    def r_squared(self, X, y):
        prediction = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, prediction)) # this calculate residual sum of square 
        ss_tot = sum((actual - y_mean) ** 2 for actual in y) #this calculate total sum of square how spread out the value in data about mean
        return 1 - (ss_res / ss_tot) #this calculate r^2 value 


model = LinearRegression(learning_rate=0.005)
model.fit(X, y, epochs=1000, print_every=200)
print(f"\nLearned: y = {model.w:.4f}x + {model.b:.4f}")
print(f"True:    y = {TRUE_W}x + {TRUE_B}")
print(f"R-squared: {model.r_squared(X, y):.4f}")






        

        
    

        


        


