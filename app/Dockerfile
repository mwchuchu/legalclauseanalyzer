# Dockerfile
FROM python:3.11-slim

# Set workdir
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --default-timeout=100 --retries=10 -r requirements.txt

# Copy only necessary app files
COPY . .    