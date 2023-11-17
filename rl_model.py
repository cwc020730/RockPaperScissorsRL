import torch
import torch.nn as nn
import torch.optim as optim

class RPSModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RPSModel, self).__init__()
        # Define the architecture of the network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Example initialization
    input_size = 2  # This can be adjusted based on how you define the state
    hidden_size = 64  # Number of neurons in the hidden layer
    output_size = 3  # Rock, Paper, Scissors

    model = RPSModel(input_size, hidden_size, output_size)
    print(model)