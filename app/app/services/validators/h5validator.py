import h5py
from fastapi import File, UploadFile


def validate_h5(file: UploadFile, fips_code: int | str) -> str:
    fips_code = str(fips_code)
    try:
        with h5py.File(file.file, "r") as f:
            if fips_code not in f.keys():
                return (
                    f"Заданного {fips_code=} нет в файле "
                    f"или неправильная структура файла"
                )
            dates = f[fips_code]
            months_days = ["-".join(date.split("-")[1:3]) for date in dates]
            if "08-15" not in months_days:
                return (
                    "Нет данных за август 15 или неправильная структура файла"
                )
    except Exception as e:
        return f"Unexpected error when reading a file: {str(e)}"

    return "ok"


if __name__ == "__main__":

    class fake:
        file = "/home/misha/code/crop-yield-prediction/data/raw/Sentinel/2019/IA/Agriculture_19_IA_2019-07-01_2019-09-30.h5"

    print(
        validate_h5(
            file=fake,
            fips_code=19001,
        )
    )
