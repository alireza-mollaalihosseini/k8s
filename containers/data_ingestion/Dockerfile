# Base image: Official Python slim (small, secure)
FROM python:3.12-slim

# Working directory inside container
WORKDIR /app

# Copy requirements + install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY app.py .

# Environment variables (learn: configurable without rebuild)
ENV DATA_DIR=/data
ENV LATITUDE=52.52
ENV LONGITUDE=13.41
ENV DAYS=30

# Volume for persistent data (optional – mounts host folder)
VOLUME /data

# Run script on container start
CMD ["python", "app.py"]