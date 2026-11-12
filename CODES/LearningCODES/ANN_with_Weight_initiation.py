import torch
import torch.nn as nn


class ANN(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(3,8)
        self.fc2=nn.Linear(8,4)
        self.fc3=nn.Linear(4,2)
        
        self.relu=nn.ReLU()
        

        # Initialize weights and biases
        self.initialize_weights()

    def initialize_weights(self):
        
        nn.init.kaiming_normal(
            self.fc1.weight,
            nonlinearity=self.relu
        )
        
        nn.init.kaiming_normal(
            self.fc2.weight,
            nonlinearity=self.relu
        )
        
        
        nn.init.xavier_uniform(self.fc3.weight)
        
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        
        
    def forward(self, x):

        x = self.relu(self.fc1(x))

        x = self.relu(self.fc2(x))

        x = self.fc3(x)

        return x


# Create model
model = ANN()

# Input: 3 features
X = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 4.0]
])

# Forward pass
output = model(X)

print("Output:\n", output)

# Count trainable parameters
total_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Total trainable parameters:", total_params)