import gc
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

# Константы
PATH_DATA = Path("data/raw")
PATH_INTERIM = Path("data/interim")
PATH_SENTINEL = PATH_DATA / "Sentinel"
PATH_HRRR = PATH_DATA / "WRF-HRRR"
PATH_ERA5 = PATH_DATA / "ERA5-Land-Moisture"
PATH_USDA = PATH_DATA / "USDA"

# Словари для переименования столбцов
COLUMN_RENAMING = {
    "HRRR": {
        "Year": "year",
        "Month": "month",
        "Day": "day",
        "FIPS Code": "fips",
        "Lat (llcrnr)": "lat_lower_left",
        "Lon (llcrnr)": "lon_lower_left",
        "Lat (urcrnr)": "lat_upper_right",
        "Lon (urcrnr)": "lon_upper_right",
        "Avg Temperature (K)": "temperature_avg",
        "Max Temperature (K)": "temperature_max",
        "Min Temperature (K)": "temperature_min",
        "Precipitation (kg m**-2)": "precipitation",
        "Relative Humidity (%)": "humidity_relative",
        "Wind Gust (m s**-1)": "wind_gust",
        "Wind Speed (m s**-1)": "wind_speed",
        "U Component of Wind (m s**-1)": "wind_u_component",
        "V Component of Wind (m s**-1)": "wind_v_component",
        "Downward Shortwave Radiation Flux (W m**-2)": "solar_radiation_downward",
        "Vapor Pressure Deficit (kPa)": "vapor_pressure_deficit",
    },
    "ERA5": {
        "src": "skin_reservoir_content",
        "swvl1": "soil_water_vol_layer1",
        "swvl2": "soil_water_vol_layer2",
        "swvl3": "soil_water_vol_layer3",
    },
    "USDA": {
        "year": "year",
        "fips": "fips",
        "YIELD, MEASURED IN BU / ACRE": "yield_bu_per_acre",
    },
}

# Целевые штаты и культуры для анализа
# TARGET_STATES = ["IOWA", "ILLINOIS", "INDIANA", "KENTUCKY", "MISSOURI", "OHIO"]
# TARGET_STATES_SHORT = ["IA", "IL", "IN", "KY", "MO", "OH"]
TARGET_STATES = ["IOWA", "ILLINOIS"]
TARGET_STATES_SHORT = ["IA", "IL"]
TARGET_CROPS = ["CORN"]


def integrate_datasets() -> None:
    """
    Основная функция для интеграции всех наборов данных.
    Выполняет загрузку, обработку и сохранение данных.
    """
    # Загрузка и подготовка данных
    hrrr_files = get_data_files(
        PATH_HRRR, states=TARGET_STATES_SHORT, crops=None
    )
    era5_files = get_data_files(
        PATH_ERA5, states=TARGET_STATES_SHORT, crops=None
    )
    usda_files = get_data_files(
        PATH_USDA, crops=[crop.lower() for crop in TARGET_CROPS], states=None
    )

    hrrr_df = prepare_hrrr(hrrr_files)
    era5_df = prepare_era5(era5_files)
    usda_df = prepare_usda(usda_files, TARGET_STATES, TARGET_CROPS)

    X = hrrr_df
    y = usda_df

    # Очистка памяти
    del hrrr_df, era5_df
    gc.collect()

    # Интеграция изображений
    sentinel_files = get_data_files(
        PATH_SENTINEL, states=TARGET_STATES_SHORT, crops=None
    )
    # Добавление путей к изображениям и сортировка
    generate_image_paths(X)
    # сохранение изображений в data/interim/images
    paths_images = prepare_and_save_sentinel(sentinel_files, X)
    # удаляем данные, для которых нет изображений
    X = clean_data_by_images(X, paths_images)

    X.sort_values(["year", "fips", "month", "day"], inplace=True)
    y.sort_values(["year", "fips"], inplace=True)

    # Сохранение данных
    save_data(X, y)
    print("Интеграция данных успешно завершена")


def prepare_hrrr(file_paths: list[Path]) -> pd.DataFrame:
    """
    Подготавливает данные WRF-HRRR.

    Args:
        file_paths (list[Path]): Список путей к файлам с данными WRF-HRRR

    Returns:
        pd.DataFrame: Обработанный DataFrame с данными WRF-HRRR
    """
    # Словарь для агрегации данных
    aggregation = {
        col: "mean"
        for col in [
            "lat_lower_left",
            "lon_lower_left",
            "lat_upper_right",
            "lon_upper_right",
            "temperature_avg",
            "temperature_max",
            "temperature_min",
            "precipitation",
            "humidity_relative",
            "wind_gust",
            "wind_speed",
            "wind_u_component",
            "wind_v_component",
            "solar_radiation_downward",
            "vapor_pressure_deficit",
        ]
    }
    aggregation.update(
        {
            "lat_lower_left": "min",
            "lon_lower_left": "min",
            "lat_upper_right": "max",
            "lon_upper_right": "max",
        }
    )

    # Колонки для удаления
    columns_to_drop = ["State", "County", "Grid Index", "Daily/Monthly"]

    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)

        df = df.dropna().drop(columns=columns_to_drop)

        df.rename(columns=COLUMN_RENAMING["HRRR"], inplace=True)

        df = (
            df.groupby(["year", "fips", "month", "day"])
            .agg(aggregation)
            .reset_index()
        )
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)

    tmp = ["year", "fips", "month", "day"]
    result[tmp] = result[tmp].astype(np.int32)
    result = result[tmp + np.sort(result.columns.difference(tmp)).tolist()]
    return result


def prepare_era5(file_paths: list[Path]) -> pd.DataFrame:
    """
    Подготавливает данные ERA5-Land.

    Args:
        file_paths (list[Path]): Список путей к файлам с данными ERA5-Land

    Returns:
        pd.DataFrame: Обработанный DataFrame с данными ERA5-Land
    """
    columns_to_drop = ["hour", "state", "latitude", "longitude"]

    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        df = df[df["day"].isin([1, 15])].dropna().drop(columns=columns_to_drop)
        grouped = (
            df.groupby(["fips", "year", "month", "day"]).mean().reset_index()
        )
        dfs.append(grouped)

    result = pd.concat(dfs, ignore_index=True)
    result.rename(columns=COLUMN_RENAMING["ERA5"], inplace=True)
    result[["year", "fips", "month", "day"]] = result[
        ["year", "fips", "month", "day"]
    ].astype(np.int32)
    return result


def prepare_usda(
    file_paths: list[Path], states: list[str], crops: list[str]
) -> pd.DataFrame:
    """
    Подготавливает данные USDA (таргетная переменная).

    Args:
        file_paths (list[Path]): Список путей к файлам с данными USDA
        states (list[str]): Список штатов для фильтрации
        crops (list[str]): Список культур для фильтрации

    Returns:
        pd.DataFrame: Обработанный DataFrame с данными USDA
    """
    df = read_multiple_csvs(file_paths)
    df["fips"] = create_fips_code(df)

    df = df[
        (df["state_name"].isin(states)) & (df["commodity_desc"].isin(crops))
    ]

    columns_to_drop = [
        "reference_period_desc",
        "state_ansi",
        "state_name",
        "county_ansi",
        "county_name",
        "asd_code",
        "asd_desc",
        "domain_desc",
        "source_desc",
        "agg_level_desc",
        "PRODUCTION, MEASURED IN BU",
    ]

    if len(crops) == 1:
        columns_to_drop.append("commodity_desc")

    df = df.drop(columns=columns_to_drop)
    df["year"] = df["year"].astype(np.int32)
    df.rename(columns=COLUMN_RENAMING["USDA"], inplace=True)
    return df


def concat_images(images: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Объединяет набор снимков в одно изображение согласно координатам. Меняет порядок следование (H, W, C) на (C, H, W)

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


def prepare_and_save_sentinel(
    file_paths: list[Path], X: pd.DataFrame
) -> list[str]:
    """
    Обрабатывает изображения Sentinel и сохраняет их в требуемом формате. Снимки для пары year, fips объединяются в один в соответствие с координатами и сохраняются как (C, H, W). Т.е. (n, 224, 224, 3) -> (H, W, 3), где H, W могут меняться от набора снимка к набору.

    Args:
        file_paths (list[Path]): Список путей к файлам с изображениями
        X (pd.DataFrame): DataFrame колонкой 'images'

    Returns:
        list[str]: Список из относительных путей к файлам изображений
    """
    existing_images_in_X = X["images"].dropna()
    images_dir = PATH_INTERIM / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    images_paths = []
    for file in file_paths:
        with h5py.File(file, "r") as h5:
            for fips, dates in h5.items():
                for date, images in dates.items():
                    file_name = f"{fips}-{date}.npy"
                    if (
                        f"images/{file_name}"
                        not in existing_images_in_X.values
                    ):
                        continue

                    output_file = images_dir / file_name
                    images_paths.append(f"images/{file_name}")
                    if output_file.exists():
                        continue

                    image = concat_images(
                        images["data"][:], images["coordinates"]
                    )
                    np.save(output_file, image)
    return images_paths


def clean_data_by_images(
    data: pd.DataFrame, paths_images: pd.Series
) -> pd.DataFrame:
    """Удаляет данные, для которых нет изображений

    Args:
        data (pd.DataFrame): датасет (должен содержать колонку images)
        paths_images (pd.Series): пути к изображениям формата images/<fips>-<year>-<month>-<day>

    Returns:
        pd.DataFrame: измененный dataframe

    """
    # paths_images format: images/<fips>-<year>-<month>-<day>
    existing_images_in_X = data["images"].dropna()
    mask = existing_images_in_X.isin(paths_images)
    needed = data.loc[mask[mask].index, ["year", "fips", "day"]]
    needed = needed.loc[
        needed.groupby(["year", "fips"])["day"].transform("count") == 24,
        ["year", "fips"],
    ].drop_duplicates()
    data = pd.merge(data, needed, how="inner", on=["year", "fips"])

    return data


def get_data_files(
    directory: Path, states: list[str] | None, crops: list[str] | None
) -> list[Path]:
    """
    Получает все файлы данных из подкаталогов указанной директории.

    Args:
        directory (Path): Путь к директории с данными
        states (list[str]): Фильтрация по штатам (аббревиатуры заглавными)
        crop_type (list[str]): Фильтрация по культуре зерна (заглавными)

    Returns:
        list[Path]: Список путей к файлам данных
    """
    # год
    states = [] if states is None else states
    crops = [] if crops is None else crops

    paths = []
    for d in directory.iterdir():
        # штат или культура
        for sd in d.iterdir():
            if sd.name not in (states + crops):
                continue
            paths.extend([f for f in sd.iterdir()])
    return paths


def read_multiple_csvs(files: list[Path]) -> pd.DataFrame:
    """
    Читает несколько CSV-файлов с одинаковой структурой в один DataFrame.

    Args:
        files (list[Path]): Список путей к CSV-файлам

    Returns:
        pd.DataFrame: Объединенный DataFrame
    """
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def create_fips_code(df: pd.DataFrame) -> pd.Series:
    """
    Создает FIPS-код из кодов штата и округа.

    Args:
        df (pd.DataFrame): DataFrame с колонками state_ansi и county_ansi

    Returns:
        pd.Series: Серия с FIPS-кодами
    """
    state_code = df["state_ansi"].astype(str).str.zfill(2)
    county_code = df["county_ansi"].astype(str).str.zfill(3)
    return (state_code + county_code).astype(np.int32)


def generate_image_paths(df: pd.DataFrame) -> None:
    """
    Генерирует относительные пути к изображениям в формате 'images/<fips>-<year>-<month>-<day>.npy'
    для строк, где day равен 1 или 15. Модифицирует входной DataFrame, добавляя колонку 'images'.

    Args:
        df (pd.DataFrame): DataFrame с колонками fips, year, month, day
    """
    df["images"] = None
    mask = (df["day"] == 1) | (df["day"] == 15)
    split = df.loc[mask]
    df.loc[mask, "images"] = (
        "images/"
        + split["fips"].astype(str)
        + "-"
        + split["year"].astype(str)
        + "-"
        + split["month"].astype(str).str.zfill(2)
        + "-"
        + split["day"].astype(str).str.zfill(2)
        + ".npy"
    )


def save_data(X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> None:
    """
    Сохраняет обработанные данные в CSV-файлы.

    Args:
        X (pd.DataFrame): DataFrame с признаками
        y (pd.DataFrame | pd.Series): Целевая переменная
    """
    if not PATH_INTERIM.exists():
        PATH_INTERIM.mkdir()
    X.to_csv(PATH_INTERIM / "X.csv", index=False)
    y.to_csv(PATH_INTERIM / "y.csv", index=False)


if __name__ == "__main__":
    integrate_datasets()
