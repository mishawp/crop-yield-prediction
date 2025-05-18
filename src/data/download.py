import calendar
import cdsapi
from pathlib import Path

PATH_ERA5 = Path("/data/raw/ERA5-Land-Moisture")
PATH_ERA5.mkdir(parents=True, exist_ok=True)  # Создаем директорию, если ее нет


def download_swvl():
    """Скачивание данных о влажности почвы из ERA5-Land"""
    client = cdsapi.Client()
    dataset = "reanalysis-era5-land"
    request = {
        "variable": [
            "skin_reservoir_content",
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3",
        ],
        "year": ["2017", "2018", "2019", "2020", "2021", "2022"],
        "month": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        # "day": [str(day) for day in range(1, 32)], # слишком большой запрос, API не разрешает
        # 1, 15 - числа съемок Sentinel, остальные для более точной оценки
        "day": ["01", "08", "15", "23"],
        "time": ["11:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [43.502, -96.640, 36.971, -87.496],  # North, West, South, East
    }

    # Для месяцев с 31 днем
    file_name = "2017-2022-IA-IL.nc"
    client.retrieve(dataset, request).download(str(PATH_ERA5 / file_name))


if __name__ == "__main__":
    download_swvl()
