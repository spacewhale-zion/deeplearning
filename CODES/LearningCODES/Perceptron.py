
import numpy as np

class Perceptron:
    
    def __init__ (self,learning_rate=0.1,epoch=100):
        self.lr=learning_rate
        self.epoch=epoch
        
    def activation(self,z):
        return 0 if z<0 else 1
    
    def fit(self,X,y):
        n_samples,n_features=X.shape
        
        self.weights=np.zeros(n_features)
        self.bias=0
        
        for epoch in range(self.epoch):
            errors=0
            
            for i in range(n_samples):
                z=np.dot(X[i],self.weights)+self.bias
                
                y_pred=self.activation(z)
                
                error= y[i]-y_pred
                
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error
                
                if error!=0:
                    errors+=1
                    
            if errors == 0:
                break
            
        return self
    
    def predict(self,X):
        
        predictions=[]
        
        for sample in X:
            z=np.dot(sample,self.weights)+self.bias
            predictions.append(self.activation(z))
            
        return np.array(predictions)
        