import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

PATH_INTERIM = Path("data/interim")
PATH_PROCESSED = Path("data/processed")


def process() -> None:
    """Основная функция обработки данных, выполняющая последовательность шагов."""
    # 1. Загрузка данных
    X = pd.read_csv(PATH_INTERIM / "X.csv")
    y = pd.read_csv(PATH_INTERIM / "y.csv")

    # 2. Добавление target_year
    X["target_year"] = np.where(X["month"] >= 11, X["year"] + 1, X["year"])

    # 3. Удаление данных первого и последнего года
    min_year, max_year = X["year"].min(), X["year"].max()
    X = filter_extreme_years(X, min_year, max_year)

    # 4. Удаление лишних признаков
    columns_to_drop = [
        "lat_lower_left",
        "lon_lower_left",
        "lat_upper_right",
        "lon_upper_right",
        "temperature_max",
        "temperature_avg",
    ]
    X.drop(columns_to_drop, axis=1, inplace=True)

    # 5. Соединение с таргетами
    data = merge_with_targets(X, y)

    # 6. Сортировка данных
    data = sort_data(data)

    # 7. Обработка пропущенных значений
    data = handle_missing_values(data)

    # 8. Удаление сентября-октября
    data = data[~data["month"].isin([9, 10])]

    # 9. Добавление средних значений таргета за предыдущий год
    data["mean_prev_year_target"] = get_prev_target_mean(data, y)

    # 10. Разделение на train/test
    X_train, X_test, y_train, y_test = split_train_test(data)

    # 11. Нормализация данных
    features_to_scale = X_train.select_dtypes(
        include=[np.float32, np.float64]
    ).columns.tolist()
    X_train, X_test, scaler = scale_features(
        X_train, X_test, features_to_scale, scaler=MinMaxScaler()
    )

    # 112. Проверка и сохранение данных
    validate_and_save(X_train, X_test, y_train, y_test)


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
    y["year"] = y["year"].astype(X["target_year"].dtype)
    data = pd.merge(
        X.drop("year", axis=1),
        y,
        how="left",
        left_on=["target_year", "fips"],
        right_on=["year", "fips"],
    )
    data.drop("year", axis=1, inplace=True)
    data.rename({"target_year": "year"}, axis=1, inplace=True)
    return data


def sort_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Сортировка данных в правильном порядке.

    Args:
        data (pd.DataFrame): DataFrame для сортировки

    Returns:
        pd.DataFrame: Отсортированный DataFrame
    """
    data["month_priority"] = np.where(data["month"] < 11, True, False)
    data.sort_values(
        ["year", "fips", "month_priority", "month", "day"], inplace=True
    )
    data.drop("month_priority", axis=1, inplace=True)
    return data


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Обработка пропущенных значений.

    Args:
        data (pd.DataFrame): DataFrame с пропущенными значениями

    Returns:
        pd.DataFrame: DataFrame с обработанными пропусками
    """
    # Удаление строк, где для января-октября нет yield_bu_per_acre
    data = data[~((data["month"] < 11) & (data["yield_bu_per_acre"].isna()))]

    # Заполнение пропусков обратным заполнением
    data.loc[:, ["year", "yield_bu_per_acre"]] = data[
        ["year", "yield_bu_per_acre"]
    ].bfill()

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
    y_year_mean = y.drop("fips", axis=1).groupby("year").mean().squeeze()
    y_year_mean.index = y_year_mean.index.astype(X["year"].dtype)
    return X["year"].apply(lambda x: y_year_mean[x - 1])


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


def validate_and_save(
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

    # Проверка целостности данных
    def check_data_integrity(data: pd.DataFrame) -> None:
        grouped = data.groupby(["year", "fips"])
        assert (grouped["yield_bu_per_acre"].nunique() == 1).all()
        assert (grouped["yield_bu_per_acre"].count() == 20).all()

    data_train = pd.concat([X_train, y_train], axis=1)
    data_test = pd.concat([X_test, y_test], axis=1)
    check_data_integrity(data_train)
    check_data_integrity(data_test)

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
