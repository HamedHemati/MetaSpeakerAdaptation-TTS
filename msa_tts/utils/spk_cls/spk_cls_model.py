import torch
import torch.nn as nn


class SpkCLSModel(nn.Module):
    def __init__(self, emb_size, hidden_size, num_cls):
        super().__init__()
        self.num_cls = num_cls
        
        self.linear1 = nn.Linear(emb_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_cls)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.softmax(self.linear2(x))
        return x