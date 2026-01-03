FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY app.py .
COPY lstm_weather.pt .
COPY weather_20260103.csv .

CMD ["python", "app.py"]