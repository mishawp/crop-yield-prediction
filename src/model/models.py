import torch
from torch import nn
from typing import Literal
from torchvision.models import resnet18


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
        out_fc = self.fc(out_rnn)

        return out_fc


class MultiCNNGRU(nn.Module):
    def __init__(self, num_frames=12, hidden_size=128, num_layers=1):
        super().__init__()

        self.num_frames = num_frames

        # Создаем отдельную CNN для каждого фрейма
        self.cnns = nn.ModuleList(
            [
                nn.Sequential(
                    *list(resnet18().children())[:-1],
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                for _ in range(num_frames)
            ]
        )

        # RNN часть
        self.rnn = nn.GRU(
            input_size=512,  # Размер фичей ResNet18
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Регрессионная головка
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x.shape = [batch_size, timesteps=12, C, H, W]
        batch_size = x.size(0)

        # Обрабатываем каждый фрейм своей CNN
        cnn_outputs = []
        for i in range(self.num_frames):
            frame = x[:, i, :, :, :]  # Берем i-й фрейм
            cnn_out = self.cnns[i](frame)  # Обрабатываем i-й CNN
            cnn_outputs.append(cnn_out)

        # Объединяем выходы CNN
        r_in = torch.stack(cnn_outputs, dim=1)  # [batch_size, timesteps, 512]

        # Пропускаем через RNN
        r_out, _ = self.rnn(r_in)

        # Берем последнее скрытое состояние
        output = self.fc(r_out[:, -1, :])

        return output


class ResNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        # Используем ResNet18 без последнего слоя
        self.resnet = nn.Sequential(
            *list(resnet18().children())[:-1],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Регрессионная головка
        self.fc = nn.Linear(512, 1)  # 512 - размер фичей ResNet18

    def forward(self, x):
        # x.shape = [batch_size, C, H, W] - одно изображение
        features = self.resnet(x)
        output = self.fc(features)
        return output
