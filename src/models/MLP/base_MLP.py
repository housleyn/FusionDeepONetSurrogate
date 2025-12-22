import torch
import torch.nn as nn
from ..activations import RowdyActivation

class BaseMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_features))
        self.activations.append(RowdyActivation(hidden_features))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.activations.append(RowdyActivation(hidden_features))
            self.dropouts.append(nn.Dropout(dropout))
        self.final = nn.Linear(hidden_features, out_features)