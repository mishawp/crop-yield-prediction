import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, efficientnet_b0
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
            input_size=512,
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
        num_frames: int = 12,
        rnns_input_size: int = None,
        rnns_hidden_size: int = 64,
        rnns_num_layers: int = 2,
        rnns_dropout: int = 0.3,
        rnns_num_last_frames: int = 7,
        main_hidden_size: int = 1024,
        main_num_layers: int = 1,
        main_dropout: int = 0.3,
        device: str = "cuda",
    ):
        super(MultiModalModel, self).__init__()
        self.device = device

        self.num_frames = num_frames

        self.rnns_num_last_frames = rnns_num_last_frames
        main_input_size = 512 + rnns_hidden_size * rnns_num_last_frames

        main_hidden_size = main_input_size * 2

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

        self.rnns = nn.ModuleList(
            [
                nn.GRU(
                    input_size=rnns_input_size,
                    hidden_size=rnns_hidden_size,
                    num_layers=rnns_num_layers,
                    dropout=rnns_dropout,
                    batch_first=True,
                )
                for _ in range(num_frames)
            ]
        )

        self.main_rnn = nn.GRU(
            input_size=main_input_size,
            hidden_size=main_hidden_size,
            num_layers=main_num_layers,
            dropout=main_dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(main_hidden_size, 1)

    def forward(self, tabular_data, image_data):
        # x.shape = [batch_size, timesteps=12, C, H, W]

        # Обрабатываем каждый фрейм своей CNN
        outputs = []
        for i in range(self.num_frames):
            frame_image = image_data[:, i, :, :, :]  # Берем i-й фрейм
            cnn_out = self.cnns[i](frame_image)  # Обрабатываем i-й CNN

            frame_tabular = tabular_data[:, i, :, :]  # Берем i-й фрейм

            rnn_out, _ = self.rnns[i](frame_tabular)
            rnn_out = rnn_out[:, -self.rnns_num_last_frames :, :]

            rnn_out = rnn_out.reshape(rnn_out.size(0), -1)

            frame_out = torch.cat((cnn_out, rnn_out), dim=1)

            outputs.append(frame_out)

        r_in = torch.stack(outputs, dim=1)

        r_out, _ = self.main_rnn(r_in)

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


class FlexibleResNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        # Базовый ResNet без последнего слоя и без глобального пулинга
        self.resnet = nn.Sequential(*list(resnet18().children())[:-2])

        # Адаптивный пулинг вместо фиксированного
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Регрессионная головка
        self.fc = nn.Linear(512, 1)  # 512 - размер фичей ResNet18

    def forward(self, x):
        # x.shape = [batch_size, C, H, W] - произвольный размер
        features = self.resnet(x)

        # Применяем адаптивный пулинг к картам признаков любого размера
        pooled = self.adaptive_pool(features)

        flattened = pooled.view(pooled.size(0), -1)
        output = self.fc(flattened)
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
