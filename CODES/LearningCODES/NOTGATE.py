from Perceptron import Perceptron
import numpy as np
# Training data
X = np.array([
   0,1
])

# Actual outputs
y = np.array([1,0])

model=Perceptron(learning_rate=0.1,epoch=200)

model.fit(X,y)

predictions=model.predict(X)

print(predictions)