import gc
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

PATH_DATA = Path("data/raw")
PATH_INTERIM = Path("data/interim")
PATH_SENTINEL = PATH_DATA / "Sentinel"
PATH_HRRR = PATH_DATA / "WRF-HRRR"
PATH_ERA5 = PATH_DATA / "ERA5-Land-Moisture"
PATH_USDA = PATH_DATA / "USDA"

RENAME_COLS_HRRR = {
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
}

RENAME_COLS_ERA5 = {
    "src": "skin_reservoir_content",
    "swvl1": "soil_water_vol_layer1",
    "swvl2": "soil_water_vol_layer2",
    "swvl3": "soil_water_vol_layer3",
}

RENAME_COLS_USDA = {
    "year": "year",
    "fips": "fips",
    "YIELD, MEASURED IN BU / ACRE": "yield_bu_per_acre",
}


def get_all_files(path: Path) -> list[Path]:
    """Возвращает все пути к файлам данных из подкаталогов

    Args:
        path (Path): путь к каталогу данных

    Returns:
        list[Path]: список путей к файлам
    """
    return [
        file
        for dir_ in path.iterdir()
        for subdir in dir_.iterdir()
        for file in subdir.iterdir()
    ]


def read_csv_files(files: list[Path]) -> pd.DataFrame:
    """Читает файлы .csv одной структуры из списка файлов files
    и возвращает их как pd.DataFrame

    Args:
        files (list[Path]): список файлов

    Returns:
        pd.DataFrame: DataFrame
    """
    return pd.concat(
        [pd.read_csv(file) for file in files], axis=0, ignore_index=True
    )


def get_fips(df_usda: pd.DataFrame) -> pd.Series:
    """Получает fips используя поля state_ansi и county_ansi

    Args:
        df_usda (pd.DataFrame): USDA dataset

    Returns:
        pd.Series: fips codes
    """
    states_fips = df_usda["state_ansi"].astype(str).str.zfill(2)
    counties_fips = df_usda["county_ansi"].astype(str).str.zfill(3)
    fips = states_fips + counties_fips
    return fips


def prepare_wrf_hrrr(paths: list[Path]) -> pd.DataFrame:
    """Подготавливает данные data/raw/WRF-HRRR для дальнейшей конкатенации

    Args:
        paths (list[Path]): пути к файлам .csv

    Returns:
        pd.DataFrame: подготовленный dataframe
    """
    agg_dict = {
        column: "mean"
        for column in [
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
    agg_dict["Lat (llcrnr)"] = "min"
    agg_dict["Lon (llcrnr)"] = "min"
    agg_dict["Lat (urcrnr)"] = "max"
    agg_dict["Lon (urcrnr)"] = "max"

    dfs_list = [None] * len(paths)
    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        df = df[(df["Day"] == 1) | (df["Day"] == 15)]

        # see ../../notebooks/1.3-data-review-wrf-hrrr.ipynb
        df.dropna(axis=0, inplace=True)
        df.drop(
            [
                "State",
                "County",
                "Grid Index",
                "Daily/Monthly",
            ],
            axis=1,
            inplace=True,
        )

        df = (
            df.groupby(["Year", "FIPS Code", "Month", "Day"])
            .agg(agg_dict)
            .reset_index()
        )

        dfs_list[i] = df

    df_full = pd.concat(dfs_list, ignore_index=True)
    df_full.rename(RENAME_COLS_HRRR, inplace=True, axis=1)
    df_full[["year", "fips", "month", "day"]] = df_full[
        ["year", "fips", "month", "day"]
    ].astype(np.int32)
    return df_full


def prepare_era5(paths: list[Path]) -> pd.DataFrame:
    """Подготавливает данные data/raw/ERA5-Land-Moisture для дальнейшей конкатенации

    Args:
        paths (list[Path]): пути к файлам .csv

    Returns:
        pd.DataFrame: подготовленный dataframe
    """
    dfs_list = [None] * len(paths)
    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        df = df[(df["day"] == 1) | (df["day"] == 15)]

        df.dropna(axis=0, inplace=True)
        df.drop(
            ["hour", "state", "latitude", "longitude"], axis=1, inplace=True
        )

        df = df.groupby(["fips", "year", "month", "day"]).mean().reset_index()

        dfs_list[i] = df

    df_full = pd.concat(dfs_list, ignore_index=True)

    df_full.rename(RENAME_COLS_ERA5, inplace=True, axis=1)

    df_full[["year", "fips", "month", "day"]] = df_full[
        ["year", "fips", "month", "day"]
    ].astype(np.int32)

    return df_full


def prepare_usda(
    paths: list[Path], states: list[str], commandity_desc: list[str]
) -> pd.DataFrame:
    """Подготавливает данные data/raw/USDA. Таргетная переменная
    Args:
        paths (list[Path]): пути к файлам .csv
        states (list[str]): штаты (полные имена заглавными буквами), для которых нужны данные
        commandity_desc (list[str]): культуры (заглавными буквами), для которых нужны данные. Доступны `CORN`

    Returns:
        pd.DataFrame: подготовленный dataframe
    """
    df = read_csv_files(paths)

    df["fips"] = get_fips(df).astype(np.int32)

    df = df[
        (df["state_name"].isin([state.upper() for state in states]))
        & (df["commodity_desc"].isin(commandity_desc))
    ]

    df.drop(
        [
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
        ],
        axis=1,
        inplace=True,
    )

    if len(commandity_desc) == 1:
        df.drop(
            "commodity_desc",
            axis=1,
            inplace=True,
        )

    df["year"] = df["year"].astype(np.int32)
    df.rename(RENAME_COLS_USDA, inplace=True, axis=1)
    return df


def prepare_sentinel(paths: list[Path], X: pd.DataFrame) -> pd.Series:
    """Подготавливает данные data/raw/Sentinel.
    - Снимки не обрабатываются, если для них нет данных из X.
    - Из X удаляются те данные, для которых нет снимков

    Args:
        paths (list[Path]): пути к файлам .h5
        X (pd.DataFrame): dataset, содержащий столбец `images` с путями к снимкам

    Returns:

    """
    path_images = PATH_INTERIM / "images"
    if not path_images.exists():
        path_images.mkdir(parents=True)

    for file in paths:
        with h5py.File(file, "r") as h5:
            for fips, attrs0 in h5.items():
                for date, attrs1 in attrs0.items():
                    dir_name = f"{fips}-{date}"
                    if "images/" + dir_name not in X["images"].values:
                        continue

                    new_dir = path_images / dir_name
                    if new_dir.exists():
                        continue

                    new_dir.mkdir()
                    for i, image in enumerate(attrs1["X"][:]):
                        np.save(new_dir / f"{i}.npy", image)


def make_images_paths(df: pd.DataFrame) -> pd.Series:
    """Создает относительные пути к изображениям формата `images/<fips>-<year>-<month>-<day>

    Args:
        df (pd.DataFrame): Dataframe, содержащий столбцы `fips`, `year`, `month` и `day`

    Returns:
        pd.Series: пути к изображениям
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
    )


def save(X: pd.DataFrame, y: pd.DataFrame | pd.Series):
    X.to_csv(PATH_INTERIM / "X.csv", index=False)
    y.to_csv(PATH_INTERIM / "y.csv", index=False)


def integrate() -> None:
    files_hrrr = get_all_files(PATH_HRRR)
    files_era5 = get_all_files(PATH_ERA5)
    files_usda = get_all_files(PATH_USDA)

    df_hrrr = prepare_wrf_hrrr(files_hrrr)
    df_era5 = prepare_era5(files_era5)
    df_usda = prepare_usda(
        files_usda,
        ["IOWA", "ILLINOIS", "INDIANA", "KENTUCKY", "MISSOURI", "OHIO"],
        ["CORN"],
    )

    X = pd.merge(
        df_hrrr, df_era5, how="inner", on=["year", "month", "day", "fips"]
    )
    y = df_usda

    del df_hrrr, df_era5
    gc.collect()

    X["images"] = make_images_paths(X)

    X.sort_values(by=["year", "fips", "month", "day"], inplace=True)
    y.sort_values(by=["year", "fips"], inplace=True)

    save(X, y)

    print("Integration completed")


if __name__ == "__main__":
    integrate()
