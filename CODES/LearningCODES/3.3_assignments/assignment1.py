import numpy as np

X=np.array([[1],[2]])
W=np.array([[0.5,0.3],[0.2,0.4]])
b=np.array([[[0.1],[-0.1]]])

Z=np.dot(W,X)+b

def RelU(z):
    return np.maximum(0,z)

A=RelU(Z)
print(A)

