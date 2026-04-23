import torch
from torch import nn

class Neural_Network(nn.Module):
    def __init__(self, input_size, layer_sizes:list):
        super().__init__()

        layers = []
        current_size = input_size
        for i in range(len(layer_sizes)):
            
            next_size = layer_sizes[i]
            layers.append(nn.Linear(current_size, next_size))

            if i < len(layer_sizes) - 1:
                layers.append(nn.ReLU())
                current_size = layer_sizes[i]
        
        self.nn_stack = nn.Sequential(*layers)
    
    def convert_to_tensors(self, X_scaled, Y_scaled):
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_scaled, dtype=torch.long).view(-1)  # returns shape (N, )
        return X_tensor, Y_tensor
    
    def forward(self, X):
        logits = self.nn_stack(X)
        return logits