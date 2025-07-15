FROM python:3.12-slim
WORKDIR /var/www
EXPOSE 5000

ENV FLASK_APP=src/app
ENV FLASK_RUN_PORT 5000
ENV FLASK_RUN_HOST=0.0.0.0

COPY requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y wget build-essential
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab
RUN python -m nltk.downloader stopwords