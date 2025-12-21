FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app

#RUN apt update -y && apt install awscli -y

#RUN apt-get update && pip install -r requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir awscli

CMD ["python3", "app.py"]

