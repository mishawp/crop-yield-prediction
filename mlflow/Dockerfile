# Используем официальный образ Python с указанием точной версии
FROM python:3.12.9

# Создаем и переходим в рабочую директорию
WORKDIR /app

# Сначала копируем только requirements.txt для лучшего кэширования слоев
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /tmp/* /var/tmp/*

# Копируем остальные файлы проекта
COPY . .

# Указываем порт, который будет использоваться
EXPOSE 5000
