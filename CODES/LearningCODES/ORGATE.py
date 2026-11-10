from Perceptron import Perceptron
import numpy as np
# Training data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Actual outputs
y = np.array([0, 1, 1, 1])

model=Perceptron(learning_rate=0.1,epoch=200)

model.fit(X,y)

predictions=model.predict(X)

print(predictions)