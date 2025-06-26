[YouTube](https://www.youtube.com/watch?v=9ff4xU9zHGw)
## 1. Создание бакета в [YC Object Storage](https://console.yandex.cloud/)

1. Создаем бакет
2. Создаем сервисный аккаунт
3. Настраиваем ACL (настроить права доступа для сервисного аккаунта)
4. Создаем API-key

## 2. Настройка dvc remote

В корне проекта делаем:
- `pip install dvc[s3]`
- `git init`
- `dvc init`
- `dvc remote add -d myremote s3://<bucket>/` [info](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3)
- `dvc remote modify myremote endpointurl <url>` [info](https://dvc.org/doc/user-guide/project-structure/configuration)
	- *Yandex Object Store* https://storage.yandexcloud.net
- `dvc remote modify --local myremote access_key_id <access_key>`
- `dvc remote modify --local myremote secret_access_key <secret_key>`

## 3. Версионирование

- `dvc add data`
	- получаем `data.dvc`
- `dvc push`
	- пушим в s3
- `dvc pull`
	- получаем версию данных, соответствующих `data.dvc`
