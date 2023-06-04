FROM python:3.9.7-slim

WORKDIR /app/
RUN apt-get update && \
    apt-get install -y build-essential libpython3.9-dev wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y build-essential python3.9-dev libpq-dev

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 4006
ENTRYPOINT uwsgi --ini uwsgi.ini
