import torch
import torch.nn as nn
import torch.optim as optim

class RPSModel(nn.Module):
    def __init__(self):
        super(RPSModel, self).__init__()
        self.relu = nn.ReLU()
        # Define the architecture of the network
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

if __name__ == '__main__':
    pass