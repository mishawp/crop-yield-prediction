import logging
import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from pathlib import Path


# Настраиваем базовую конфигурацию логирования
logging.basicConfig(
    # Устанавливаем уровень логирования DEBUG
    level=logging.DEBUG,
    # Задаем формат сообщений лога
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class Preprocessing:
    transforms_compose = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(512),  # Ресайз до 512px по меньшей стороне
            transforms.CenterCrop(512),  # Кроп до точного размера 512x512
            transforms.ToTensor(),  # Преобразование в тензор [0,1]
            transforms.Normalize(  # Нормализация по ImageNet
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    @classmethod
    def preprocess(cls, file: Path) -> torch.Tensor:
        """
        Предобработка npz файла. Чтение, объединение снимков, изменение размера и нормализация.

        Args:
            file (Path): путь к файлу

        Returns:
            np.ndarray: изображение
        """
        images, coordinates = cls._get_images(file)
        image = cls._concat_images(images, coordinates)
        return cls._resize(image)

    @staticmethod
    def _get_images(file: Path) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает набор снимков и их координат

        Args:
            file (Path): путь к файлу

        Returns:
            tuple[np.ndarray, np.ndarray]: набор снимков, координаты
        """
        data = np.load(file)
        images = data["images"]
        coordinates = data["coordinates"]
        return (images, coordinates)

    @staticmethod
    def _concat_images(
        images: np.ndarray, coordinates: np.ndarray
    ) -> np.ndarray:
        """
        Объединяет набор снимков в одно изображение согласно координатам. Меняет порядок следование (H, W, C) на (C, H, W)

        Args:
            images (np.ndarray): Набор снимков формы (n, height, width, channels).
            coordinates (np.ndarray): Координаты снимков формы (n, 2, 2),
                                    где [i, 0, :] — нижний левый угол (lat, lon),
                                    [i, 1, :] — верхний правый угол (lat, lon).

        Returns:
            np.ndarray: Объединённое изображение формы
                        (n_rows * height, n_cols * width, channels).
        """
        ll_coords = coordinates[:, 0, :]  # Нижние левые углы (lat, lon)

        # Уникальные широты и долготы (сортировка для правильного расположения)
        # Широты: большие сверху
        unique_lats = np.sort(np.unique(ll_coords[:, 0]))[::-1]
        # Долготы: малые слева
        unique_lons = np.sort(np.unique(ll_coords[:, 1]))

        n_rows = len(unique_lats)
        n_cols = len(unique_lons)
        n_images, height, width, channels = images.shape

        # Создаём пустое изображение для сборки
        combined_image = np.zeros(
            (n_rows * height, n_cols * width, channels), dtype=images.dtype
        )

        # Заполняем сетку изображениями
        for img_idx in range(n_images):
            ll_lat, ll_lon = ll_coords[img_idx]
            # Находим строку и столбец для текущего изображения
            row = np.where(unique_lats == ll_lat)[0][0]
            col = np.where(unique_lons == ll_lon)[0][0]

            # Вычисляем координаты вставки
            y_start = row * height
            y_end = (row + 1) * height
            x_start = col * width
            x_end = (col + 1) * width

            # Вставляем изображение
            combined_image[y_start:y_end, x_start:x_end, :] = images[img_idx]

        return np.transpose(combined_image, (2, 0, 1))

    @classmethod
    def _resize(cls, image: np.ndarray) -> torch.Tensor:
        """
        Предобработка снимка

        Args:
            image (pd.Series): снимок
        """

        # Конвертируем в тензор и ресайзим
        image_t = torch.from_numpy(image)
        return cls.transforms_compose(image_t)


class EfficientNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = efficientnet_b0()
        self.features = self.efficientnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1280, 1)

        self.eval()
        logger.info(f"Model inited {self.__class__.__name__}")

    def forward(self, x):
        # x.shape = [batch_size, 3, 512, 512]
        features = self.features(x)  # [batch_size, 1280, H', W']
        pooled = self.avgpool(features)  # [batch_size, 1280, 1, 1]
        flattened = self.flatten(pooled)  # [batch_size, 1280]
        output = self.fc(flattened)  # [batch_size, 1]
        return output


class Predictor:
    MODELS = Path("models")

    def __init__(
        self,
        preprocessor: Preprocessing,
        model: nn.Module,
        state_dict: dict | None = None,
        device: str = "cuda",
    ):
        self.preprocessor = preprocessor
        self.device = device if torch.cuda.is_available() else "cpu"
        torch.device(self.device)

        if state_dict is not None:
            model.load_state_dict(state_dict)

        self.model = model.to(self.device)

        logger.info(
            f"Predictor inited. Model {model.__class__.__name__}, "
            f"device {self.device}"
        )

    def predict(self, file: Path) -> float:
        processed_data = self.preprocessor.preprocess(file)
        with torch.no_grad():
            result = self.model(processed_data.unsqueeze(0).to(self.device))
        return result.squeeze(0).item()


predictor = Predictor(
    Preprocessing(),
    EfficientNetRegressor(),
    torch.load(
        Predictor.MODELS / "EfficientNetRegressor_20250602_1702_r2_0.6516.pth",
        map_location="cpu",
        weights_only=True,
    ),
    "cuda",
)
