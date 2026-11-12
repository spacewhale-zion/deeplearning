
import torch
import torch.nn as nn


# -----------------------------
# 1. Dataset
# -----------------------------

torch.manual_seed(42)

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])


# -----------------------------
# 2. Define neural network
# -----------------------------

class XORNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(4, 1)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):

        x = self.fc1(x)
        x = self.tanh(x)

        # Return raw logits
        x = self.fc2(x)

        return x


# -----------------------------
# 3. Model, loss and optimizer
# -----------------------------

model = XORNet()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.03
)


# -----------------------------
# 4. Training
# -----------------------------

epochs = 5000

for epoch in range(epochs):

    # Clear old gradients
    optimizer.zero_grad()

    # Forward propagation
    logits = model(X)

    # Calculate loss
    loss = criterion(logits, y)

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    if epoch % 1000 == 0:
        print(
            f"Epoch: {epoch}, "
            f"Loss: {loss.item():.6f}"
        )


# -----------------------------
# 5. Final predictions
# -----------------------------

model.eval()

with torch.no_grad():

    logits = model(X)

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= 0.5).int()

    print("\nProbabilities:")
    print(probabilities)

    print("\nPredictions:")
    print(predictions)
