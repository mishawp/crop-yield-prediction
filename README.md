# Baseline

## Структура проекта

\* - данных нет в репозитории github и нужны только для отчетов. 

```plaintext
├── data/                                       <- Корневая директория для всех данных проекта
│   │
│   ├── external/                               <- Внешние данные, не изменяемые в рамках проекта (например, справочники)
│   │   │
│   │   └── state_and_county_fips_master.csv    <- Мастер-файл с кодами FIPS штатов и округов
│   │
│   ├── interim/                                <- Промежуточные данные, полученные в процессе обработки
│   │
│   ├── processed/                              <- Окончательные обработанные данные, готовые для обучения
│   │
│   └── raw/                                    <- Исходные сырые данные (не изменяются)
│       │
│       ├── ERA5-Land-Moisture/                 <- Данные о влажности почвы из ERA5-Land
│       │
│       ├── Sentinel/                           <- Данные спутника Sentinel
│       │
│       ├── USDA/                               <- Исторические данные об урожайности США
│       │
│       └── WRF-HRRR/                           <- Метеорологические данные модели WRF-HRRR
│
├── notebooks/                                  <- Jupyter notebooks для анализа и визуализации
│   │
│   ├── 1.1-data-review-sentinel.ipynb          <- Обзор и предварительный анализ данных Sentinel
│   │
│   ├── 1.2-data-review-usda.ipynb              <- Обзор данных USDA
│   │
│   ├── 1.3-data-review-wrf-hrrr.ipynb          <- Анализ метеорологических данных
│   │
│   ├── 1.4-data-review-moisture.ipynb          <- Анализ данных о влажности почвы
│   │
│   └── 2.0-EDA.ipynb                           <- Notebook для разведочного анализа данных (EDA)
│
├── obsidian/                                   <- Заметки в формате Obsidian
│
├── resources/                                  <- * Ресурсы проекта (изображения, карты и т.д.)
│   │
│   └── USA-map/                                <- Географические данные по США
│
├── src/                                        <- Исходный код проекта
│   │
│   ├── data/                                   <- Скрипты для работы с данными
│   │   │
│   │   ├── download.py                         <- Скрипты для загрузки данных
│   │   │
│   │   ├── ERA5_nc_to_csv.ipynb                <- Конвертация данных ERA5 из netCDF в CSV
│   │   │
│   │   └── integration.ipynb                   <- Интеграция данных из разных источников
│
├── .dvcignore                                  <- Файл исключений для DVC (аналог .gitignore)
│
├── .gitignore                                  <- Файл исключений для Git
│
├── data.dvc                                    <- Файл конфигурации DVC для управления данными
│
├── dvc.lock                                    <- Файл блокировки DVC (аналог package-lock.json)
│
├── dvc.yaml                                    <- Конфигурация pipelines DVC
│
├── README.md                                   <- Основная документация проекта
│
├── requirements.txt                            <- Зависимости Python
│
└── .env                                        <- Переменные окружения (чувствительные данные)
```

## Примечания к текущей версии (baseline)

[Спутниковые снимки Sentinel](notebooks/1.1-data-review-sentinel.ipynb) не используются. Только табличные данные. Поэтому, если надо воспроизвести обучение, то `data/raw/Sentinel` загружать не надо. Делаем

- `dvc pull data/raw/USDA data/raw/ERA5-Land-Moisture data/raw/WRF-HRRR`
- `dvc repro`

После этих шагов, помимо воспроизведения pipeline, появятся файлы, оканчивающиеся на `.nbconvert.ipynb` - Jupiter Notebooks с ходом выполнения.

## Навигация по проекту

### 1. Описание данных

- [спутниковые снимки Sentinel](notebooks/1.1-data-review-sentinel.ipynb);
- [исторические данные об урожайности](notebooks/1.2-data-review-usda.ipynb);
- [климатические данные](notebooks/1.3-data-review-wrf-hrrr.ipynb);
- [данные о влажности почвы](notebooks/1.4-data-review-moisture.ipynb).

### 2. Интеграция данных

Соединение датасетов воедино

- [Интеграция данных](/src/data/integration.ipynb).

### 3. Исследовательский анализ данных

Анализ данных, полученных после *интеграции*

- [EDA](notebooks/2.0-EDA.ipynb)
