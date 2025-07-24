import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.15) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.fc1 = nn.Linear(
            in_features=self.input_dim,  out_features=self.hidden_dim, bias=True)
        self.fc2 = nn.Linear(
            in_features=self.hidden_dim, out_features=self.output_dim, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        return hidden