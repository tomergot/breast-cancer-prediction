import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class BoostedMetastasisNN(nn.Module):
    def __init__(self, input_dim, dropout=0.3, hidden_dims=(64, 32), output_dim=11):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_nn(X_boosted, y_train, device, hidden_dims=(64, 32), n_epochs=10, dropout=0.3,
             batch_size=128, lr=1e-3):
    model = BoostedMetastasisNN(input_dim=X_boosted.shape[1], dropout=dropout,
                                hidden_dims=hidden_dims).to(device)
    X_tensor = torch.tensor(X_boosted, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    print(X_tensor.device, y_tensor.device)

    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(loader):.4f}")
    return model
