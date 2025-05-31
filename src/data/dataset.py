import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

PATH_PROCESSED = Path("data/processed")


class TabularDataset(Dataset):
    def __init__(self, path_X: Path, path_y: Path):
        X = pd.read_csv(path_X)
        y = pd.read_csv(path_y)

        data = pd.concat([X, y], axis=1)

        # Группируем по year и fips, сохраняя группы как списки записей
        grouped = data.groupby(["year", "fips"])
        self.n_samples = len(grouped)

        # Собираем группы в списки
        X_groups = [None] * self.n_samples
        y_groups = [None] * self.n_samples

        for i, ((year, fips), group) in enumerate(grouped):
            # Отбрасываем ненужные столбцы для X
            # images пока не используем
            X_values = group.drop(
                ["month", "day", "yield_bu_per_acre", "images"], axis=1
            ).values

            # для пары fips, year только одно значение yield_bu_per_acre
            # условие проверяется в data/process.ipynb
            y_values = group["yield_bu_per_acre"].iloc[0]

            images_values = group["images"].values

            X_groups[i] = X_values
            y_groups[i] = y_values

        # Преобразуем списки в numpy массивы
        self.X = np.array(X_groups)  # 3D array: (sample, timestep, features)
        self.y = np.array(y_groups)  # 1D array: (target value per sample)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]).float(),
            torch.tensor(self.y[idx]).float(),
        )


class ImagesDataset(Dataset):
    def __init__(self, path_X: Path, path_y: Path):
        X = pd.read_csv(path_X, usecols=["year", "fips", "images"])
        y = pd.read_csv(path_y)

        data = pd.concat([X, y], axis=1).dropna()

        # Группируем по year и fips, сохраняя группы как списки записей
        grouped = data.groupby(["year", "fips"])
        self.n_samples = len(grouped)

        # Собираем группы в списки
        X_groups = [None] * self.n_samples
        y_groups = [None] * self.n_samples

        for i, ((year, fips), group) in enumerate(grouped):
            X_values = group["images"].values

            # для пары fips, year только одно значение yield_bu_per_acre
            y_values = group["yield_bu_per_acre"].iloc[0]

            X_groups[i] = X_values
            y_groups[i] = y_values

        # Преобразуем списки в numpy массивы
        self.X = np.array(X_groups)  # 2D array: (sample, timestep)
        self.y = np.array(y_groups)  # 1D array: (target value per sample)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        paths_images = PATH_PROCESSED / self.X[idx]
        images = np.array([np.load(path) for path in paths_images])
        return (
            torch.tensor(images).float(),
            torch.tensor(self.y[idx]).float(),
        )


class OneImageDataset(Dataset):
    def __init__(self, path_X: Path, path_y: Path):
        X = pd.read_csv(
            path_X, usecols=["year", "fips", "month", "day", "images"]
        )
        y = pd.read_csv(path_y)

        data = pd.concat([X, y], axis=1).dropna()
        data = data[(data["month"] == 6) & (data["day"] == 15)]
        self.X = data["images"].values
        self.y = data["yield_bu_per_acre"].values
        self.n_samples = self.X.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        path_image = PATH_PROCESSED / self.X[idx]
        image = np.load(path_image)
        return (
            torch.tensor(image).float(),
            torch.tensor(self.y[idx]).float(),
        )
