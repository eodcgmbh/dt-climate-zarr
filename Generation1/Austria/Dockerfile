FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ClimateDT_Austria.py .
COPY src/desp-authentication.py ./src/desp-authentication.py
