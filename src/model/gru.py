import torch
from torch import nn


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 200,
        num_layers: int = 2,
        dropout: float = 0.3,
        device="cuda",
    ):
        """
        GRU model for regression tasks
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units
            num_layers (int): Number of GRU layers
            dropout (float): Dropout probability
            device (str): Device to run on ('cuda' or 'cpu')
        """
        super(GRURegressor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Output layer for regression (single output)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            X: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, 1)
        """
        # Initial hidden state
        h_0 = torch.zeros(
            self.num_layers,
            X.size(0),
            self.hidden_size,
        ).to(self.device)

        # GRU output
        out_gru, _ = self.gru(X, h_0)

        # Take the last timestep output
        out_gru = out_gru[:, -1, :]

        # Final regression output
        out_fc = self.fc(out_gru)

        return out_fc
