import numpy as np
import argparse
import h5py
from pathlib import Path


class DataNotExist(Exception):
    pass


def process_data(
    input_file: Path, fips_code: str, date: str
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает набор снимков и их координат

    Args:
        input_file (Path): входной файл
        fips_code (str): fips код
        date (str): дата (yyyy-mm-dd)

    Returns:
        tuple[np.ndarray, np.ndarray]: набор снимков, координаты
    """

    with h5py.File(input_file, "r") as f:
        if fips_code not in f.keys():
            raise DataNotExist(f"Файл не содержит данных для fips {fips_code}")
        dates = f[fips_code]
        if date not in dates.keys():
            raise DataNotExist(f"Файл не содержит данных на {date}")
        images = dates[date]
        return (images["data"][:], images["coordinates"][:])


def main():
    parser = argparse.ArgumentParser(
        description="Обработка данных по FIPS и дате"
    )
    parser.add_argument("input_file", help="Путь к входному файлу")
    parser.add_argument("fips_code", help="FIPS code")
    parser.add_argument("date", help="Дата в формате yyyy-mm-dd")
    parser.add_argument(
        "output_dir", help="Директория для сохранения результата"
    )

    args = parser.parse_args()

    images, coordinates = process_data(
        args.input_file, args.fips_code, args.date
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{args.fips_code}-{args.date}.npz"

    np.savez(output_file, images=images, coordinates=coordinates)
    print(f"Результат успешно сохранен в {output_file}")


if __name__ == "__main__":
    main()

# python processor.py input.h5 output.npz
