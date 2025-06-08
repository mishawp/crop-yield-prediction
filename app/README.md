# Прогнозирование урожайности сельскохозяйственных культур

## Структура проекта

├── app/                          <- Основное приложение (API + веб-интерфейс)
│   ├── api.py                    <- Главный файл API (FastAPI)
│   ├── auth/                     <- Модуль аутентификации
│   ├── database/                 <- Работа с базой данных
│   ├── models/                   <- Модели данных (ORM)
│   ├── routes/                   <- Маршруты API
│   ├── services/                 <- Бизнес-логика приложения
│   ├── tests/                    <- Тесты
│   └── view/                     <- HTML шаблоны
│
├── ml_worker/                    <- Отдельный сервис для ML задач
│   ├── model.py                  <- ML модель
│   └── rmq/                      <- Работа с RabbitMQ (очереди задач)
│
├── nginx/                        <- Конфигурация Nginx
│   └── nginx.conf
│
├── docker-compose.yaml           <- Конфигурация Docker Compose
├── README.md                     <- Описание проекта
└── utils/                        <- Вспомогательные скрипты
    └── extract_images.py

## Инструкция по запуску

- Модель принимает на вход набор изображений и соответствующих им координат за Август 15 дня;
- данные для проверки работоспособности API
  - можно скачать .h5 из [Yandex Disk](https://disk.yandex.ru/d/0_xYpNxARedQVQ);
  - [скриптом](/utils/extract_images.py) получить из `.h5` набор изображений и координат в формате `<fips>-<year>-<month>-<day>.npz`. Команда имеет следующий вид `python extract_images.py path/to/.h5 fips_code yyyy-mm-dd output/dir`. Напр., `python extract_images.py Agriculture_19_IA_2019-07-01_2019-09-30.h5 19001 2019-08-15 ../images`.

- Запустить API можно выполнив следующие команды
  - `docker-compose build`
  - `docker-compose up`
- Замечания
  - Приложение использует модель из `../models`, в частности (может меняться) `../models/EfficientNetRegressor_20250602_1702_r2_0.6516.pth`. Получить его можно, сделав `dvc pull models`. Только вот ключи ... Но можно скачать из [Yandex Disk](https://disk.yandex.ru/d/2gSUkvjc23b8yw)

- [Инструкция по настройке cuda](../obsidian/Instructions/Cuda.md)
- 