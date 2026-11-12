
import torch
import torch.nn as nn

class ANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(4, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Hidden layer
        Z1 = self.fc1(x)
        A1 = self.relu(Z1)

        # Output layer
        Z2 = self.fc2(A1)
        A2 = self.sigmoid(Z2)

        print("Hidden Layer Weighted Sum (Z1):")
        print(Z1)

        print("\nHidden Layer Activation (A1):")
        print(A1)

        print("\nOutput Layer Weighted Sum (Z2):")
        print(Z2)

        print("\nOutput Layer Activation (A2):")
        print(A2)

        return A2


# Create model
model = ANN()

# Input with 3 features
X = torch.tensor([[2.0, 1.0, 3.0]])

# Forward propagation
output = model(X)

print("\nFinal Prediction:")
print(output)
