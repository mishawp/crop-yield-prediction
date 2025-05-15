import cdsapi


# download_swvl_except_feb и download_swvl_feb
# скачивают файлы с одинаковыми названиями
def download_swvl_except_feb():
    """Скачивание данных о влажности почвы из
    [ERA5-Land hourly data from 1950 to present]
    (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?).
    Кроме февраля.
    """
    dataset = "reanalysis-era5-land"
    request = {
        "variable": [
            "skin_reservoir_content",
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3",
        ],
        "year": ["2018", "2019", "2020", "2021"],
        "month": [
            "01",
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
        "day": ["01", "15", "30"],
        "time": ["11:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        # North, West, South, East
        "area": [43.502, -96.640, 36.971, -87.496],
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()


def download_swvl_feb():
    """Скачивание данных о влажности почвы из
    [ERA5-Land hourly data from 1950 to present]
    (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?).
    Только февраль.
    """
    dataset = "reanalysis-era5-land"
    request = {
        "variable": [
            "skin_reservoir_content",
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3",
        ],
        "year": ["2018", "2019", "2020", "2021"],
        "month": ["02"],
        "day": ["01", "15", "28"],
        "time": ["11:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        # North, West, South, East
        "area": [43.502, -96.640, 36.971, -87.496],
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()
