FROM python:3.12-slim
WORKDIR /var/www
ENV FLASK_APP=src/app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN apt-get update && apt-get install -y wget build-essential
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000