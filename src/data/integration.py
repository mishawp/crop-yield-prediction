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
TARGET_STATES = ["IOWA", "ILLINOIS"]
TARGET_STATES_SHORT = ["IA", "IL"]
TARGET_CROPS = ["CORN"]


def integrate_datasets() -> None:
    """
    Основная функция для интеграции всех наборов данных.
    Выполняет загрузку, обработку и сохранение данных.
    """
    # Загрузка и подготовка данных
    sentinel_files = get_data_files(
        PATH_SENTINEL, states=TARGET_STATES_SHORT, crops=None
    )
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

    # Объединение признаков
    X = pd.merge(
        hrrr_df, era5_df, how="inner", on=["year", "month", "day", "fips"]
    )
    y = usda_df

    # Очистка памяти
    del hrrr_df, era5_df
    gc.collect()

    # Интеграция изображений
    # Добавление путей к изображениям и сортировка
    X["images"] = generate_image_paths(X)
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
    Подготавливает данные WRF-HRRR для интеграции.

    Args:
        file_paths (list[Path]): Список путей к файлам с данными WRF-HRRR

    Returns:
        pd.DataFrame: Обработанный DataFrame с данными WRF-HRRR
    """
    # Словарь для агрегации данных
    aggregation = {
        col: "mean"
        for col in [
            "Lat (llcrnr)",
            "Lon (llcrnr)",
            "Lat (urcrnr)",
            "Lon (urcrnr)",
            "Avg Temperature (K)",
            "Max Temperature (K)",
            "Min Temperature (K)",
            "Precipitation (kg m**-2)",
            "Relative Humidity (%)",
            "Wind Gust (m s**-1)",
            "Wind Speed (m s**-1)",
            "U Component of Wind (m s**-1)",
            "V Component of Wind (m s**-1)",
            "Downward Shortwave Radiation Flux (W m**-2)",
            "Vapor Pressure Deficit (kPa)",
        ]
    }
    aggregation.update(
        {
            "Lat (llcrnr)": "min",
            "Lon (llcrnr)": "min",
            "Lat (urcrnr)": "max",
            "Lon (urcrnr)": "max",
        }
    )

    # Колонки для удаления
    columns_to_drop = ["State", "County", "Grid Index", "Daily/Monthly"]

    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        df = (
            df[(df["Day"].isin([1, 15]))]
            .dropna()
            .drop(columns=columns_to_drop)
        )

        grouped = (
            df.groupby(["Year", "FIPS Code", "Month", "Day"])
            .agg(aggregation)
            .reset_index()
        )
        dfs.append(grouped)

    result = pd.concat(dfs, ignore_index=True)
    result.rename(columns=COLUMN_RENAMING["HRRR"], inplace=True)
    result[["year", "fips", "month", "day"]] = result[
        ["year", "fips", "month", "day"]
    ].astype(np.int32)
    return result


def prepare_era5(file_paths: list[Path]) -> pd.DataFrame:
    """
    Подготавливает данные ERA5-Land для интеграции.

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


def prepare_and_save_sentinel(
    file_paths: list[Path], X: pd.DataFrame
) -> list[str]:
    """
    Обрабатывает изображения Sentinel и сохраняет их в требуемом формате.

    Args:
        file_paths (list[Path]): Список путей к файлам с изображениями
        X (pd.DataFrame): DataFrame колонкой 'images'

    Returns:
        list[str]: Список из относительных путей к файлам изображений
    """
    images_dir = PATH_INTERIM / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    images_paths = []
    for file in file_paths:
        with h5py.File(file, "r") as h5:
            for fips, dates in h5.items():
                for date, images in dates.items():
                    file_name = f"{fips}-{date}.npy"
                    if f"images/{file_name}" not in X["images"].values:
                        continue

                    output_file = images_dir / file_name
                    images_paths.append(f"images/{file_name}")
                    if output_file.exists():
                        continue

                    # (X, 224, 224, 3) - X -кол
                    np.save(output_file, images["data"][:])
    return images_paths


def clean_data_by_images(data: pd.DataFrame, paths_images: pd.Series):
    out = data[data["images"].isin(paths_images)]
    mask = out.groupby(["year", "fips"])["day"].transform("count") == 24
    images_for_del = out.loc[~mask, "images"]
    for path in images_for_del:
        (PATH_INTERIM / path).unlink()
    out = out[mask]
    return out


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


def generate_image_paths(df: pd.DataFrame) -> pd.Series:
    """
    Генерирует относительные пути к изображениям в формате 'images/<fips>-<year>-<month>-<day>'.

    Args:
        df (pd.DataFrame): DataFrame с колонками fips, year, month, day

    Returns:
        pd.Series: Серия с путями к изображениям
    """
    return (
        "images/"
        + df["fips"].astype(str)
        + "-"
        + df["year"].astype(str)
        + "-"
        + df["month"].astype(str).str.zfill(2)
        + "-"
        + df["day"].astype(str).str.zfill(2)
        + ".npy"
    )


def save_data(X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> None:
    """
    Сохраняет обработанные данные в CSV-файлы.

    Args:
        X (pd.DataFrame): DataFrame с признаками
        y (pd.DataFrame | pd.Series): Целевая переменная
    """
    X.to_csv(PATH_INTERIM / "X.csv", index=False)
    y.to_csv(PATH_INTERIM / "y.csv", index=False)


if __name__ == "__main__":
    integrate_datasets()
