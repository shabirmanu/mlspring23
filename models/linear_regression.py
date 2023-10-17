import numpy as np
class LinearRegression():
    def __init__(self, lr = 0.01, iters=1000):
        self.lr = lr
        self.iters = iters
        self.weight = None
        self.bias = None
        
    
    def fit(self, X, y):
        

        X = _scale_features(X)
        samples , features = X.shape

        self.weight = np.zeros(features)
        #self.bias = 0
        
        # w = 1, b =2 => y_pred = wx+b is our hypothesis
        # 
        for _ in range(self.iters):
            y_pred = np.dot(X, self.weight)

            absoluter_err = y_pred - y
            # error = 1/2 (y - y_pred)**2
            # if(error < self.breakrule):
            #     return
            ## Gradient descent rule 

            ## dw (E) = (y - y_pred)dw(y_pred)
            # => dw(y_pred) = y - (wx + b)x
            ## db (E) = (y - y_pred)db(y_pred)
            # => dw(y_pred) = y - (wx + b) = 1
            
            
            dq = np.dot(X.T, absoluter_err)
            #db = np.sum(absoluter_err)

            self.weight = self.weight - self.lr*dq
            #self.bias = self.bias - self.lr*db
    
    def predict(self, X):
        return np.dot(X, self.weight)



def _scale_features(X):
    _ , features = X.shape
    x0 = np.ones(features)
    return np.insert(X, 0, x0, axis=1)