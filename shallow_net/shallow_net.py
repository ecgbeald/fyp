from torch import nn


class ShallowNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class ShallowMultiLabelNet(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(ShallowMultiLabelNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_labels)
        self.out = nn.Sigmoid()  # sigmoid for multi-label

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        return self.out(self.fc2(x))  # shape: (batch_size, num_labels)
