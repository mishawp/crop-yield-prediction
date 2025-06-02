import torch
from torch import nn
from typing import Literal
from torchvision.models import resnet18, efficientnet_b0


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
        RNN model for regression tasks that uses last 7 timesteps
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units
            num_layers (int): Number of RNN layers
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
        self.num_last_frames = 7  # Number of last frames to consider

        self.rnn = RNNRegressor.rnn_types[rnn_type](
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Output layer for regression (single output)
        # Input size is now hidden_size * num_last_frames because we concatenate the last 7 frames
        self.fc = nn.Linear(hidden_size * self.num_last_frames, 1)

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

        # Take the last 7 timesteps outputs
        out_rnn = out_rnn[:, -self.num_last_frames :, :]

        # Flatten the last num_last_frames frames
        out_rnn = out_rnn.reshape(out_rnn.size(0), -1)

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


class MultiModalModel(nn.Module):
    def __init__(
        self,
        rnn_type: Literal["RNN", "LSTM", "GRU"],
        tabular_input_size: int,
        image_num_frames: int = 12,
        tabular_hidden_size: int = 200,
        image_hidden_size: int = 128,
        tabular_num_layers: int = 2,
        image_num_layers: int = 1,
        dropout: float = 0.3,
        device="cuda",
    ):
        super(MultiModalModel, self).__init__()
        self.device = device

        # Модель для табличных данных
        self.tabular_model = RNNRegressor(
            rnn_type=rnn_type,
            input_size=tabular_input_size,
            hidden_size=tabular_hidden_size,
            num_layers=tabular_num_layers,
            dropout=dropout,
            device=device,
        )

        # Модель для изображений
        self.image_model = MultiCNNGRU(
            num_frames=image_num_frames,
            hidden_size=image_hidden_size,
            num_layers=image_num_layers,
        )

        # Объединяющий слой
        self.combine_fc = nn.Linear(2, 1)  # Объединяем 2 выхода в 1

    def forward(self, tabular_data, image_data):
        # Обработка табличных данных
        tabular_out = self.tabular_model(tabular_data)

        # Обработка изображений
        image_out = self.image_model(image_data)

        # Объединение результатов
        combined = torch.cat([tabular_out, image_out], dim=1)
        output = self.combine_fc(combined)

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


class EfficientNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        # Загружаем EfficientNet-B0 из torchvision
        self.efficientnet = efficientnet_b0()

        # Удаляем классификационную головку (она включает avgpool и classifier)
        # Оставляем только features (conv layers)
        self.features = self.efficientnet.features

        # Добавляем свои слои для регрессии
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Для EfficientNet-B0 выход features имеет 1280 каналов
        self.fc = nn.Linear(1280, 1)

    def forward(self, x):
        # x.shape = [batch_size, 3, 512, 512]
        features = self.features(x)  # [batch_size, 1280, H', W']
        pooled = self.avgpool(features)  # [batch_size, 1280, 1, 1]
        flattened = self.flatten(pooled)  # [batch_size, 1280]
        output = self.fc(flattened)  # [batch_size, 1]
        return output
