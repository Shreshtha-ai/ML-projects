class LinearRegression:
    def__init__(self, learning_rate = 0.01): #init initialize the object 
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

        
    

        


        


