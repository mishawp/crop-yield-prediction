import torch
from torch import nn
from typing import Literal


class RNNRegressor(nn.Module):
    rnn_types = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}

    def __init__(
        self,
        rnn_type: Literal["RNN", "LSTM", "GRU"],
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
        super(RNNRegressor, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.rnn = RNNRegressor.rnn_types[rnn_type](
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Output layer for regression (single output)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

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

        # For LSTM, we need to initialize both hidden and cell states
        if self.rnn_type == "LSTM":
            c_0 = torch.zeros(
                self.num_layers,
                X.size(0),
                self.hidden_size,
            ).to(self.device)
            out_rnn, _ = self.rnn(X, (h_0, c_0))
        else:
            out_rnn, _ = self.rnn(X, h_0)

        # Take the last timestep output
        out_rnn = out_rnn[:, -1, :]

        # Final regression output
        out_fc1 = self.fc1(out_rnn)
        out_fc2 = self.fc2(out_fc1)

        return out_fc2
