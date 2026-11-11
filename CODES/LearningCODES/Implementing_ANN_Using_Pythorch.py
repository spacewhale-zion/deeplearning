
import torch
import torch.nn as nn


class ANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# Create model
model = ANN()

# Input data
X = torch.tensor(
    [[2.0, 1.0]],
    dtype=torch.float32
)

# Forward propagation
output = model(X)

print("Prediction:", output)
