import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler

PATH_INTERIM = Path("data/interim")
PATH_PROCESSED = Path("data/processed")


def process() -> None:
    """Основная функция обработки данных, выполняющая последовательность шагов."""
    # 1. Загрузка данных
    X = pd.read_csv(PATH_INTERIM / "X.csv")
    y = pd.read_csv(PATH_INTERIM / "y.csv")

    # 2. Добавление target_year
    X = X[(X["month"] > 2) & (X["month"] < 9)]  # март - август включительно

    # 3. Удаление лишних признаков
    columns_to_drop = [
        "lat_lower_left",
        "lon_lower_left",
        "lat_upper_right",
        "lon_upper_right",
    ]
    X.drop(columns_to_drop, axis=1, inplace=True)

    # 4. Соединение с таргетами
    data = merge_with_targets(X, y)

    # 5. Сортировка данных
    data = sort_data(data)

    # 6. Обработка изображений (приведение к (3, W, H))
    resize_and_save_images(data["images"].dropna())

    # 7. Разделение на train/test
    X_train, X_test, y_train, y_test = split_train_test(data)

    # 8. Нормализация данных
    features_to_scale = X_train.select_dtypes(
        include=[np.float32, np.float64]
    ).columns.tolist()
    X_train, X_test, scaler = scale_features(
        X_train, X_test, features_to_scale, scaler=MinMaxScaler()
    )

    # Упорядочим колонки
    sort_cols = ["year", "fips", "month", "day", "images"]
    X_train = X_train[
        sort_cols + np.sort(X_train.columns.difference(sort_cols)).tolist()
    ]
    X_test = X_test[
        sort_cols + np.sort(X_test.columns.difference(sort_cols)).tolist()
    ]

    # 9. Проверка и сохранение данных
    save(X_train, X_test, y_train, y_test)


def filter_extreme_years(
    df: pd.DataFrame, min_year: int, max_year: int
) -> pd.DataFrame:
    """
    Удаление данных первого и последнего года.

    Args:
        df (pd.DataFrame): DataFrame с данными
        min_year (int): Минимальный год в данных
        max_year (int): Максимальный год в данных

    Returns:
        pd.DataFrame: Отфильтрованный DataFrame
    """
    return df[
        ~(
            ((df["year"] == min_year) & (df["month"] < 11))
            | ((df["year"] == max_year) & (df["month"] > 10))
        )
    ]


def merge_with_targets(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """
    Соединение признаков с таргетами.

    Args:
        X (pd.DataFrame): DataFrame с признаками
        y (pd.DataFrame): DataFrame с таргетами

    Returns:
        pd.DataFrame: Объединенный DataFrame
    """
    y["year"] = y["year"].astype(X["year"].dtype)
    data = pd.merge(
        X,
        y,
        how="inner",
        on=["year", "fips"],
    )
    return data


def sort_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Сортировка данных в правильном порядке.

    Args:
        data (pd.DataFrame): DataFrame для сортировки

    Returns:
        pd.DataFrame: Отсортированный DataFrame
    """
    data.sort_values(["year", "fips", "month", "day"], inplace=True)
    return data


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Обработка пропущенных значений.

    Args:
        data (pd.DataFrame): DataFrame с пропущенными значениями

    Returns:
        pd.DataFrame: DataFrame с обработанными пропусками
    """
    # Оставляем только полные годы (12 месяцев)
    data = data[
        data.groupby(["year", "fips"])["month"].transform("nunique") == 12
    ]
    return data


def get_prev_target_mean(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """
    Среднее значение таргета за предыдущий год в признаки

    Args:
        X (pd.DataFrame): DataFrame с признаками
        y (pd.DataFrame): DataFrame с таргетами

    Returns:
        pd.Series: Серия с средними значениями таргета за предыдущий год
    """
    y = y.copy()
    X = X[["year", "fips"]].copy()

    y["state"] = y["fips"].astype(str).str[:2].astype(np.int32)
    y.drop("fips", axis=1, inplace=True)

    y_year_mean = y.groupby(["year", "state"]).mean().squeeze()

    X["state"] = X["fips"].astype(str).str[:2].astype(np.int32)
    return X.apply(lambda x: y_year_mean[(x["year"] - 1, x["state"])], axis=1)


def process_images_to_features(
    paths_images: pd.Series,
    device: str = "cuda",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Извлекает усредненные признаки изображений с помощью ResNet18.

    Args:
        paths_images (pd.Series): Список путей к .npy файлам с изображениями
        device (str): Устройство для вычислений ('cuda' или 'cpu')
        show_progress (bool): Показывать прогресс-бар

    Returns:
        DataFrame с признаками
    """
    # Инициализация устройства
    device = torch.device(
        "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    )

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights).to(device)

    # Удаляем последний слой (классификатор)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # Трансформации с использованием новых весов
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=weights.transforms().mean, std=weights.transforms().std
            ),
        ]
    )

    features_list = []

    # Оборачиваем в tqdm для отображения прогресса
    iterator = (
        tqdm(paths_images, desc="Processing images")
        if show_progress
        else paths_images
    )

    for npy_file in iterator:
        # Загружаем изображения
        images = np.load(npy_file)  # shape: (n, 224, 224, 3)

        # Батч-обработка
        with torch.no_grad():
            batch = torch.stack([transform(img) for img in images]).to(device)
            features = model(batch)  # (n, 512, 1, 1)
            # Добавляем Global Average Pooling и сжимаем размерности
            features = features.mean(dim=[2, 3]).cpu().numpy()  # (n, 512)
            features = features
            avg_features = np.mean(features, axis=0)  # (512,)

        features_list.append(avg_features)

    # Создаем DataFrame
    features_df = pd.DataFrame(features_list, index=paths_images.index)
    features_df.columns = [f"embed_{i}" for i in range(features_df.shape[1])]

    return features_df


def resize_and_save_images(paths_images: pd.Series):
    """Резайзит изображения и сохраняет их по пути PATH_PROCESSED / images
    Args:
        paths_images (pd.Series): относительные пути к изображениям .npy
    """
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),  # изменение размера
            transforms.CenterCrop(224),  # центр-кроп до 224x224
            transforms.ToTensor(),  # преобразование в тензор [0,1]
            transforms.Normalize(  # нормализация по ImageNet
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    if not (tmp := (PATH_PROCESSED / "images")).exists():
        tmp.mkdir(parents=True)
    for path in paths_images:
        image = np.load(PATH_INTERIM / path)

        # Конвертируем в тензор и ресайзим
        image_t = torch.from_numpy(image)
        image_resized = preprocess(image_t)

        np.save(PATH_PROCESSED / path, image_resized)


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    features_to_scale: list[str],
    scaler: TransformerMixin = MinMaxScaler(),
) -> tuple[pd.DataFrame, pd.DataFrame, TransformerMixin]:
    """
    Масштабирует указанные признаки в обучающих и тестовых данных с помощью MinMaxScaler.

    Args:
        X_train (pd.DataFrame): Обучающий набор данных
        X_test (pd.DataFrame): Тестовый набор данных
        features_to_scale (list[str]): Список названий признаков, которые нужно масштабировать
        scaler : объект scaler, по умолчанию MinMaxScaler() Объект scaler с методами fit/transform (например, StandardScaler)

    Returns:
        tuple[DataFrame, DataFrame]: Кортеж с масштабированными версиями X_train и X_test

    """
    # Создаем копии данных чтобы избежать предупреждений
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Масштабируем только указанные признаки
    scaler.fit(X_train[features_to_scale])

    # Применяем масштабирование
    X_train_scaled = scaler.transform(X_train[features_to_scale])
    X_test_scaled = scaler.transform(X_test[features_to_scale])

    # Обновляем масштабированные признаки
    X_train[features_to_scale] = X_train_scaled
    X_test[features_to_scale] = X_test_scaled

    return X_train, X_test, scaler


def split_train_test(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разделение данных на train/test.

    Args:
        data (pd.DataFrame): Полный DataFrame для разделения

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        Кортеж с (X_train, X_test, y_train, y_test)
    """
    mask = data["year"] == data["year"].max()
    data_train = data[~mask]
    data_test = data[mask]

    X_train = data_train.drop("yield_bu_per_acre", axis=1)
    y_train = data_train["yield_bu_per_acre"]
    X_test = data_test.drop("yield_bu_per_acre", axis=1)
    y_test = data_test["yield_bu_per_acre"]

    return X_train, X_test, y_train, y_test


def save(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs,
) -> None:
    """
    Проверка и сохранение данных.

    Args:
        X_train (pd.DataFrame): Признаки для обучения
        X_test (pd.DataFrame): Признаки для тестирования
        y_train (pd.Series): Целевая переменная для обучения
        y_test (pd.Series): Целевая переменная для тестирования
        **kwargs: параметры, которые нужно сохранить через joblib (напр., MinMaxScaler())
    """
    # Упорядочивание столбцов
    columns_order = ["year", "fips", "month", "day"] + sorted(
        X_train.columns.drop(["year", "fips", "month", "day"])
    )
    X_train = X_train[columns_order]
    X_test = X_test[columns_order]

    # Сохранение данных
    if not PATH_PROCESSED.exists():
        PATH_PROCESSED.mkdir()

    X_train.to_csv(PATH_PROCESSED / "X_train.csv", index=False)
    y_train.to_csv(PATH_PROCESSED / "y_train.csv", index=False)
    X_test.to_csv(PATH_PROCESSED / "X_test.csv", index=False)
    y_test.to_csv(PATH_PROCESSED / "y_test.csv", index=False)
    for key, value in kwargs:
        joblib.dump(value, PATH_PROCESSED / f"{key}.pkl")

    # Дополнительная статистика
    X_train_size = X_train[["year", "fips"]].drop_duplicates().shape[0]
    X_test_size = X_test[["year", "fips"]].drop_duplicates().shape[0]
    print(f"X_train unique year-fips: {X_train_size}")
    print(f"X_test unique year-fips: {X_test_size}")
    print(f"X_test/X: {X_test_size / (X_test_size + X_train_size):.4f}")


if __name__ == "__main__":
    process()
