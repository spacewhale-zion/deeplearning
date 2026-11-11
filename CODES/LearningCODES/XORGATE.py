
import numpy as np
from Perceptron import Perceptron

# Inputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Train OR
y_or = np.array([0, 1, 1, 1])

or_model = Perceptron(learning_rate=0.1, epoch=100)
or_model.fit(X, y_or)

# Train AND
y_and = np.array([0, 0, 0, 1])

and_model = Perceptron(learning_rate=0.1, epoch=100)
and_model.fit(X, y_and)

# Combine predictions
or_output = or_model.predict(X)
and_output = and_model.predict(X)

# XOR = OR AND NOT(AND)
xor_output = or_output & (~and_output.astype(bool))

print("OR: ", or_output)
print("AND:", and_output)
print("XOR:", xor_output.astype(int))
