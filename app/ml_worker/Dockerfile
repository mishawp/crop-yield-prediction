FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install torchvision==0.22.0

COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install -r /app/requirements.txt

CMD ["python", "main.py"]