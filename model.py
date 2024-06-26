
import torch.nn as nn
'''
Defining the Generator class, which includes initialization and the forward pass. 
The generator uses a three-layer fully connected network with ReLU activations.
'''

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
'''
Defining the Discriminator class, which includes initialization and the forward pass.
The discriminator uses a three-layer fully connected network with ReLU activations 
and a sigmoid activation function in the output layer for binary classification.
'''
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)